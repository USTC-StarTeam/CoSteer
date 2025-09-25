from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import json
import torch.nn.functional as F
import argparse
import os

# ==========================================================================
# §1. 新增：Byte-level Greedy 解码技术相关辅助函数和类
# ==========================================================================

def _bytes_to_unicode():
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b); cs.append(256 + n); n += 1
    return dict(zip(bs, map(chr, cs)))

def _build_unicode_to_bytes():
    b2u = _bytes_to_unicode()
    return {u: b for b, u in b2u.items()}

def _token_id_to_bytes(tokenizer, tid, u2b=None):
    if tid in set(tokenizer.all_special_ids or []):
        return None
    tok = tokenizer.convert_ids_to_tokens(int(tid), skip_special_tokens=False)
    if tok is None:
        return None
    if getattr(tokenizer, "byte_decoder", None):
        dec = tokenizer.byte_decoder
        try:
            return bytes([dec.get(ch, None) if isinstance(ch, str) else ch for ch in tok])
        except Exception:
            pass
    u2b = u2b or _build_unicode_to_bytes()
    try:
        return bytes([u2b[c] for c in tok])
    except Exception:
        s = tokenizer.convert_tokens_to_string([tok])
        return s.encode("utf-8")

def _scatter_logsumexp(values: torch.Tensor, buckets: torch.Tensor, dim_size: int = 257) -> torch.Tensor:
    assert buckets.dtype == torch.long
    v = values
    device, dtype = v.device, v.dtype

    out = torch.full((dim_size,), float("-inf"), device=device, dtype=dtype)
    m = torch.full_like(out, float("-inf"))
    m.scatter_reduce_(0, buckets, v, reduce="amax", include_self=False)
    m = torch.where(torch.isinf(m), torch.zeros_like(m), m)
    s = torch.zeros_like(out)
    s.scatter_add_(0, buckets, torch.exp(v - m[buckets]))
    eps = 1e-12 if dtype == torch.float32 else (1e-6 if dtype == torch.bfloat16 else 1e-4)
    return m + torch.log(torch.clamp(s, min=eps))

class ByteGreedyHelper:
    def __init__(self, tokenizer, vocab_size, device, allow_special=True, debug=False):
        self.tok = tokenizer
        self.vocab_size = int(vocab_size)
        self.device = device
        self.allow_special = allow_special
        self.debug = debug
        u2b = _build_unicode_to_bytes()
        self._tok_bytes = [None] * self.vocab_size
        for tid in range(self.vocab_size):
            self._tok_bytes[tid] = _token_id_to_bytes(self.tok, tid, u2b)
        self._nonspecial_mask = torch.ones(self.vocab_size, dtype=torch.bool)
        for sp in (self.tok.all_special_ids or []):
            self._nonspecial_mask[int(sp)] = False
        self._kth_byte_cache = {}

    def _kth_bytes(self, k: int) -> torch.Tensor:
        t = self._kth_byte_cache.get(k)
        if t is not None:
            return t
        vec = torch.full((self.vocab_size,), 256, dtype=torch.long)
        for tid, b in enumerate(self._tok_bytes):
            if b is not None and k < len(b):
                vec[tid] = b[k]
        vec = vec.to(self.device)
        self._kth_byte_cache[k] = vec
        return vec

    def _tok_str(self, tid: int) -> str:
        s = self.tok.convert_ids_to_tokens(int(tid), skip_special_tokens=False)
        return s if s is not None else self.tok.decode([int(tid)], skip_special_tokens=False)

    @torch.inference_mode()
    def pick_token_ids(self, logprobs_last: torch.Tensor, eos_token_id=None, *, step_idx=None) -> torch.Tensor:
        lp = logprobs_last if logprobs_last.dim() == 2 else logprobs_last.unsqueeze(0)
        B, V = lp.shape
        assert V == self.vocab_size
        nonsp = self._nonspecial_mask.to(lp.device).clone()
        if eos_token_id is not None:
            nonsp[int(eos_token_id)] = True
        out = []
        for b in range(B):
            cand = nonsp.nonzero(as_tuple=False).squeeze(1)
            k = 0
            if self.debug:
                print(f"[step={step_idx or 0} sample={b}] start candidates={cand.numel()}")
            while True:
                kb = self._kth_bytes(k)[cand]
                L = _scatter_logsumexp(lp[b, cand], kb, 257)
                b_star = int(torch.argmax(L).item())
                if b_star == 256:
                    end_mask = (kb == 256)
                    if not end_mask.any():
                        chosen = int(torch.argmax(lp[b]).item())
                        reason = "end(empty)->globalMAP"
                    else:
                        pool = cand[end_mask]
                        if not self.allow_special:
                            keep = nonsp[pool]
                            pool = pool[keep] if keep.any() else pool
                        chosen = int(pool[torch.argmax(lp[b, pool])].item())
                        reason = "end(256)"
                    out.append(chosen)
                    if self.debug:
                        tstr = self._tok_str(chosen)
                        print(f"  [k={k}] choose END  -> tid={chosen} tok={tstr!r} reason={reason}")
                    break
                else:
                    keep = (kb == b_star)
                    before = int(cand.numel()); cand = cand[keep]; after = int(cand.numel())
                    if self.debug:
                        ch = chr(b_star) if 32 <= b_star < 127 else "."
                        print(f"  [k={k}] byte=0x{b_star:02X}('{ch}')  {before} -> {after}")
                    k += 1
                    if cand.numel() == 1:
                        chosen = int(cand.item())
                        out.append(chosen)
                        if self.debug:
                            tstr = self._tok_str(chosen)
                            print(f"  [k={k}] unique -> tid={chosen} tok={tstr!r}")
                        break
        return torch.tensor(out, device=lp.device, dtype=torch.long)

# ==========================================================================
# §2. 原有代码结构
# ==========================================================================

class CosteerGenerator:
    def __init__(self, T, alpha, beta, player_lambda, eta):
        self.iteration_num = T
        self.alpha = alpha
        self.beta = beta
        self.player_lambda = player_lambda
        self.eta = eta
        
    def optimize_policy(self, llm_logits, slm_wo_logits, slm_with_logits):
        batch_size, vocab_size = llm_logits.shape
        log_player = torch.log_softmax(llm_logits, dim=-1)
        log_ref = torch.log_softmax(llm_logits, dim=-1)
        slm_with_logits = torch.log_softmax(slm_with_logits, dim=-1)
        slm_wo_logits = torch.log_softmax(slm_wo_logits, dim=-1)
        Q = torch.zeros((batch_size, self.iteration_num + 1, vocab_size), device=llm_logits.device)
        log_players_0 = log_player.clone()
        log_player_mem = torch.zeros_like(Q)
        for cur_iter in range(1, self.iteration_num + 1):
            log_player_mem[:, cur_iter - 1] = log_player.detach()
            Q[:, cur_iter] = self.alpha * (log_player - log_ref) + self.beta * (slm_with_logits - slm_wo_logits)
            term1 = (cur_iter) * self.player_lambda * log_players_0
            term2 = torch.sum(Q[:, 0:cur_iter + 1], dim=1)
            term3 = log_player_mem[:, cur_iter - 1] / (self.eta)
            denominator = cur_iter * self.player_lambda + 1 / (self.eta)
            log_player = (term1 + term2 + term3) / denominator
            log_player = torch.log_softmax(log_player, dim=-1)
        return log_player

def load_models(args):
    """加载模型和分词器"""
    LLM_model = AutoModelForCausalLM.from_pretrained(
        args.llm_model_name,
        torch_dtype="auto",
        device_map="auto"
    )
    SLM_model = AutoModelForCausalLM.from_pretrained(
        args.slm_model_name,
        torch_dtype="auto",
        device_map="auto"
    )
    LLM_model.eval()
    SLM_model.eval()
    LLM_tokenizer = AutoTokenizer.from_pretrained(args.llm_model_name)
    SLM_tokenizer = AutoTokenizer.from_pretrained(args.slm_model_name)
    return LLM_model, SLM_model, LLM_tokenizer, SLM_tokenizer

def create_vocab_intersection_map(llm_tokenizer, slm_tokenizer, device):
    """
    创建 LLM 和 SLM 词汇表的交集，并返回对齐的 token ID 张量。
    """
    print("Creating vocabulary intersection map...")
    llm_vocab = llm_tokenizer.get_vocab()
    slm_vocab = slm_tokenizer.get_vocab()
    llm_tokens = set(llm_vocab.keys())
    slm_tokens = set(slm_vocab.keys())
    intersect_tokens = llm_tokens.intersection(slm_tokens)
    print(f"LLM vocab size: {len(llm_vocab)}")
    print(f"SLM vocab size: {len(slm_vocab)}")
    print(f"Vocabulary intersection size: {len(intersect_tokens)} tokens.")
    llm_ids_list = []
    slm_ids_list = []
    for token in sorted(list(intersect_tokens)):
        llm_ids_list.append(llm_vocab[token])
        slm_ids_list.append(slm_vocab[token])
    llm_intersect_ids = torch.tensor(llm_ids_list, dtype=torch.long, device=device)
    slm_intersect_ids = torch.tensor(slm_ids_list, dtype=torch.long, device=device)
    return llm_intersect_ids, slm_intersect_ids

def make_top_5_prompt(query, top_5):
    prompt_parts = ["The following are five titles with their abstracts."]
    items_to_use = top_5[:5]
    for i, item in enumerate(items_to_use):
        prompt_parts.append(f"Title[{i+1}]: {item['title']}\nAbstract[{i+1}]: {item['abstract']}\n")
    prompt_parts.append("Now it's your turn\n")
    prompt_parts.append(query)
    return "\n".join(prompt_parts)

# +++++++++++++++++++++++++++++++++++++++++++++++++++++
# +++ 重大修改：generate_response 函数以使用 Byte-level Greedy +++
# +++++++++++++++++++++++++++++++++++++++++++++++++++++
def generate_response(query_wo, query_with, LLM_model, SLM_model, LLM_tokenizer, SLM_tokenizer, 
                      llm_intersect_ids, slm_intersect_ids, llm_id_to_slm_id, byte_picker, args):
    """处理单个item的生成"""
    messages_wo = [{"role": "system", "content": "You are a helpful assistant."},
                 {"role": "user", "content": query_wo}]
    messages_with = [{"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": query_with}]

    llm_text_wo = LLM_tokenizer.apply_chat_template(
        messages_wo, tokenize=False, add_generation_prompt=True)
    slm_text_wo = SLM_tokenizer.apply_chat_template(
        messages_wo, tokenize=False, add_generation_prompt=True)
    slm_text_with = SLM_tokenizer.apply_chat_template(
        messages_with, tokenize=False, add_generation_prompt=True)

    llm_inputs = LLM_tokenizer([llm_text_wo], return_tensors="pt").to(LLM_model.device)
    slm_inputs_wo = SLM_tokenizer([slm_text_wo], return_tensors="pt").to(SLM_model.device)
    slm_inputs_with = SLM_tokenizer([slm_text_with], return_tensors="pt").to(SLM_model.device)

    max_new_tokens = args.max_new_tokens
    llm_seq = llm_inputs.input_ids
    slm_seq_wo = slm_inputs_wo.input_ids
    slm_seq_with = slm_inputs_with.input_ids

    costeer_optimizer = CosteerGenerator(T=args.T, alpha=args.alpha, beta=args.beta, 
                                      player_lambda=args.player_lambda, eta=args.eta)
    
    for _ in range(max_new_tokens):
        with torch.no_grad():
            slm_wo_logits_native = SLM_model(slm_seq_wo).logits[:, -1, :]
            slm_with_logits_native = SLM_model(slm_seq_with).logits[:, -1, :]
            llm_logits_native = LLM_model(llm_seq).logits[:, -1, :]

        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # +++ 新增：强制停止机制 (Forced Stop Mechanism)
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # 在进入Costeer和Byte-level解码前，先检查原始SLM with context的停止意图
        slm_with_probs = torch.softmax(slm_with_logits_native, dim=-1)
        slm_with_eos_prob = slm_with_probs[0, SLM_tokenizer.eos_token_id]
        
        # 如果SLM with context 输出EOS的概率超过设定的阈值，则直接中断生成
        if slm_with_eos_prob.item() > args.eos_force_threshold:
            print(f"\nINFO: SLM_with EOS probability ({slm_with_eos_prob.item():.4f}) exceeded threshold ({args.eos_force_threshold}). Forcing generation to stop.")
            break
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

        intersect_logits_llm = llm_logits_native.index_select(-1, llm_intersect_ids)
        intersect_logits_slm_wo = slm_wo_logits_native.index_select(-1, slm_intersect_ids)
        intersect_logits_slm_with = slm_with_logits_native.index_select(-1, slm_intersect_ids)
        
        combined_logits = costeer_optimizer.optimize_policy(
            intersect_logits_llm, intersect_logits_slm_wo, intersect_logits_slm_with
        )
        
        # --- Byte-level Greedy 解码 ---
        # 1. 将优化后的交集 logits 扩展回 LLM 的完整词表空间
        full_llm_logits = torch.full(
            (1, LLM_model.config.vocab_size), 
            float('-inf'), 
            device=LLM_model.device, 
            dtype=combined_logits.dtype
        )
        full_llm_logits.scatter_(-1, llm_intersect_ids.unsqueeze(0), combined_logits)
        full_llm_logprobs = F.log_softmax(full_llm_logits, dim=-1)

        # 2. 使用 Byte-level Greedy 选择器选择 token
        next_llm_token = byte_picker.pick_token_ids(
            full_llm_logprobs, 
            eos_token_id=LLM_tokenizer.eos_token_id
        ).squeeze()

        # 3. 反向映射，检查 token 是否对 SLM 有效
        next_slm_token_id = llm_id_to_slm_id.get(next_llm_token.item())
        
        if next_slm_token_id is None:
            # 如果选出的 token 不在交集中，无法在 SLM 中继续，终止生成
            print(f"Warning: Byte-greedy chose LLM token {next_llm_token.item()} ('{LLM_tokenizer.decode(next_llm_token)}') which is not in the SLM's vocabulary. Stopping generation.")
            break
            
        next_slm_token = torch.tensor([[next_slm_token_id]], device=SLM_model.device)
        
        llm_seq = torch.cat((llm_seq, next_llm_token.view(1, 1)), dim=1)
        slm_seq_wo = torch.cat((slm_seq_wo, next_slm_token), dim=1)
        slm_seq_with = torch.cat((slm_seq_with, next_slm_token), dim=1)

        # 保留原有的停止条件作为备用
        if next_llm_token.item() == LLM_tokenizer.eos_token_id:
            break

    generated_text = LLM_tokenizer.decode(
        llm_seq[0, len(llm_inputs.input_ids[0]):], 
        skip_special_tokens=True
    )
    
    return generated_text

def read_json_and_extract_info(args):
    LLM_model, SLM_model, LLM_tokenizer, SLM_tokenizer = load_models(args)
    
    llm_intersect_ids, slm_intersect_ids = create_vocab_intersection_map(LLM_tokenizer, SLM_tokenizer, LLM_model.device)
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # +++ 修改：使用 'input' 作为唯一标识来实现断点续跑 +++
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    processed_inputs = set()
    # 检查输出文件是否存在，如果存在则加载已处理的 'input' 内容
    if os.path.exists(args.output_file):
        print(f"发现已存在的输出文件: {args.output_file}。正在加载已处理的 'input'...")
        with open(args.output_file, 'r', encoding='utf-8') as f_out:
            for line in f_out:
                try:
                    processed_item = json.loads(line)
                    # 将 'input' 字段的内容加入 set 中
                    if 'input' in processed_item:
                        processed_inputs.add(processed_item['input'])
                except json.JSONDecodeError:
                    print(f"警告：输出文件中有一行无法解析，已跳过: {line.strip()}")
        print(f"加载完成，共找到 {len(processed_inputs)} 个已处理的 'input'。现在开始继续任务...")
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # +++                     修改结束                       +++
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # +++ 新增：初始化 Byte-level Greedy 解码器和反向映射表 +++
    byte_picker = ByteGreedyHelper(
        tokenizer=LLM_tokenizer,
        vocab_size=LLM_model.config.vocab_size,
        device=LLM_model.device,
        debug=False # 可设为 True 以查看详细解码过程
    )
    llm_id_to_slm_id = {llm_id.item(): slm_id.item() for llm_id, slm_id in zip(llm_intersect_ids, slm_intersect_ids)}

    with open(args.input_file, 'r', encoding='utf-8') as file:
        for line in file:
            item = json.loads(line)
            id = item.get('id')
            input = item.get('input')
            if input in processed_inputs:
                # 为了日志整洁，可以只打印 input 的前缀
                print(f"Input: '{input[:70]}...' 已处理，跳过。")
                continue # 跳到下一个循环
            top_5 = item.get('top_5')
            prompt_wo_user_profile = input
            prompt_with_user_profile = make_top_5_prompt(input, top_5)
            
            response = generate_response(prompt_wo_user_profile, prompt_with_user_profile, 
                                        LLM_model, SLM_model, LLM_tokenizer, SLM_tokenizer,
                                        llm_intersect_ids, slm_intersect_ids, 
                                        llm_id_to_slm_id, byte_picker, args)
            
            new_json = {
                "id": id,
                'input': input,
                'response': response,
            }
            with open(args.output_file, 'a', encoding='utf-8') as output_file:
                json.dump(new_json, output_file, ensure_ascii=False)
                output_file.write('\n')

def parse_args():
    parser = argparse.ArgumentParser(description="Costeer Generator with hyperparameters")
    parser.add_argument("--T", type=int, default=20, help="Number of iterations")
    parser.add_argument("--alpha", type=float, default=2, help="Alpha parameter")
    parser.add_argument("--beta", type=float, default=1, help="Beta parameter")
    parser.add_argument("--player_lambda", type=float, default=2, help="Player lambda parameter")
    parser.add_argument("--eta", type=float, default=10, help="Eta parameter")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p sampling parameter")
    parser.add_argument("--max_new_tokens", type=int, default=1024, help="Maximum number of new tokens to generate")
    parser.add_argument("--input_file", type=str, default="datasets/longlamp/abstract.jsonl", help="Path to input JSONL file")
    parser.add_argument("--output_dir", type=str, default="rebuttal/mix", help="Directory for output files")
    parser.add_argument("--llm_model_name", type=str, default="models/Llama3.1-8B-Instruct", help="Path to LLM model")
    parser.add_argument("--slm_model_name", type=str, default="models/Qwen2.5-1.5B-Instruct", help="Path to SLM model")
    parser.add_argument("--eos_force_threshold", type=float, default=0.5, help="Threshold for forcing generation to stop")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    llm_model_short = os.path.basename(args.llm_model_name)
    slm_model_short = os.path.basename(args.slm_model_name)
    
    output_filename = f"abstract_byte_slm_with_force_stop_{args.eos_force_threshold}_{llm_model_short}_{slm_model_short}_costeer_v1_{args.T}_{args.alpha}_{args.beta}_{args.player_lambda}_{args.eta}_temp{args.temperature}_p{args.top_p}_initial_log_softmax_t-1_greedy.jsonl"
    args.output_file = os.path.join(args.output_dir, output_filename)
    
    return args

if __name__ == "__main__":
    args = parse_args()
    read_json_and_extract_info(args)