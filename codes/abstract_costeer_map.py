from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import json
import torch.nn.functional as F
import argparse
import os

class CosteerGenerator:
    def __init__(self, T, alpha, beta, player_lambda, eta):
        self.iteration_num = T
        self.alpha = alpha
        self.beta = beta
        self.player_lambda = player_lambda
        self.eta = eta
        
    def optimize_policy(self, llm_logits, slm_wo_logits, slm_with_logits):
        # 维度对齐
        batch_size, vocab_size = llm_logits.shape
        
        # === 变量初始化 ===
        log_player = torch.log_softmax(llm_logits, dim=-1)  # [batch, vocab]
        log_ref = torch.log_softmax(llm_logits, dim=-1)    # [batch, vocab]
        
        slm_with_logits = torch.log_softmax(slm_with_logits, dim=-1)  # [batch, vocab]
        slm_wo_logits = torch.log_softmax(slm_wo_logits, dim=-1)    # [batch, vocab]
        
        Q = torch.zeros((batch_size, self.iteration_num + 1, vocab_size), 
                    device=llm_logits.device)
        
        log_players_0 = log_player.clone()  # 初始策略记忆体
        
        log_player_mem = torch.zeros_like(Q)  # 策略记忆体
        
        # === 迭代优化 ===
        for cur_iter in range(1, self.iteration_num+1):
            log_player_mem[:, cur_iter-1] = log_player.detach()
            Q[:, cur_iter] = self.alpha * (log_player - log_ref) + self.beta * (slm_with_logits - slm_wo_logits)
            
            # 策略更新公式
            term1 = (cur_iter) * self.player_lambda * log_players_0
            term2 = torch.sum(Q[:, 0:cur_iter+1], dim=1)
            term3 = log_player_mem[:, cur_iter-1] / (self.eta)
            
            denominator = cur_iter * self.player_lambda + 1/(self.eta)

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

# +++++++++++++++++++++++++++++++++++++++++++++
# +++ 修改：采用词表交集方式构建映射 +++
# +++++++++++++++++++++++++++++++++++++++++++++
def create_vocab_intersection_map(llm_tokenizer, slm_tokenizer, device):
    """
    创建 LLM 和 SLM 词汇表的交集，并返回对齐的 token ID 张量。
    """
    print("Creating vocabulary intersection map...")
    llm_vocab = llm_tokenizer.get_vocab()
    slm_vocab = slm_tokenizer.get_vocab()

    # 寻找共同的 token 字符串
    llm_tokens = set(llm_vocab.keys())
    slm_tokens = set(slm_vocab.keys())
    intersect_tokens = llm_tokens.intersection(slm_tokens)

    print(f"LLM vocab size: {len(llm_vocab)}")
    print(f"SLM vocab size: {len(slm_vocab)}")
    print(f"Vocabulary intersection size: {len(intersect_tokens)} tokens.")

    # 为交集创建对齐的 token ID 列表
    llm_ids_list = []
    slm_ids_list = []
    # 排序以保证每次运行的映射都是确定的
    for token in sorted(list(intersect_tokens)):
        llm_ids_list.append(llm_vocab[token])
        slm_ids_list.append(slm_vocab[token])

    # 转换为张量
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
# +++ 重大修改：generate_response 函数以使用词表交集 +++
# +++++++++++++++++++++++++++++++++++++++++++++++++++++
def generate_response(query_wo, query_with, LLM_model, SLM_model, LLM_tokenizer, SLM_tokenizer, llm_intersect_ids, slm_intersect_ids, args):
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

        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # +++ 新增：强制停止机制 (Forced Stop Mechanism) +++
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # 检查原始 SLM with context 输出 eos_token 的概率是否超过阈值
        # 这可以防止 Costeer 优化过程稀释掉 SLM with context 强烈的停止信号
        
        # 将原始 SLM with context logits 转换为概率分布
        llm_probs = torch.softmax(llm_logits_native, dim=-1)
        llm_eos_prob = llm_probs[0, LLM_tokenizer.eos_token_id]
        
        # 如果概率大于设定的阈值（例如 0.5），则强制中断循环
        # 这个阈值可以根据需要调整，甚至可以作为一个新的超参数
        if llm_eos_prob.item() > args.eos_force_threshold:
            print(f"\nINFO: LLM EOS probability ({llm_eos_prob.item():.4f}) exceeded threshold ({args.eos_force_threshold}). Forcing generation to stop.")
            break
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # +++                       修改结束                         +++
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

        # --- 词表对齐 (Intersection) ---
        intersect_logits_llm = llm_logits_native.index_select(-1, llm_intersect_ids)
        intersect_logits_slm_wo = slm_wo_logits_native.index_select(-1, slm_intersect_ids)
        intersect_logits_slm_with = slm_with_logits_native.index_select(-1, slm_intersect_ids)
        
        # 执行Costeer优化
        combined_logits = costeer_optimizer.optimize_policy(
            intersect_logits_llm, intersect_logits_slm_wo, intersect_logits_slm_with
        )
        
        # 从共同词表空间中选择下一个 token 的 *索引*
        probs = F.softmax(combined_logits, dim=-1)
        next_token_idx = torch.argmax(probs, dim=-1)
        
        # --- Token反向映射 ---
        next_llm_token = llm_intersect_ids[next_token_idx]
        next_slm_token = slm_intersect_ids[next_token_idx]
        
        # 更新序列
        llm_seq = torch.cat((llm_seq, next_llm_token.view(1, 1)), dim=1)
        slm_seq_wo = torch.cat((slm_seq_wo, next_slm_token.view(1, 1)), dim=1)
        slm_seq_with = torch.cat((slm_seq_with, next_slm_token.view(1, 1)), dim=1)

        # 保留原有的停止条件作为备用
        if next_llm_token.item() == LLM_tokenizer.eos_token_id:
            break

    generated_text = LLM_tokenizer.decode(
        llm_seq[0, len(llm_inputs.input_ids[0]):], 
        skip_special_tokens=True
    )
    
    return generated_text
    
    f

def read_json_and_extract_info(args):
    LLM_model, SLM_model, LLM_tokenizer, SLM_tokenizer = load_models(args)
    
    # +++ 修改：创建交集映射表 +++
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
                                        llm_intersect_ids, slm_intersect_ids, args) # 传递交集映射表
            
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
    
    output_filename = f"abstract_mix_force_stop_{args.eos_force_threshold}_{llm_model_short}_{slm_model_short}_costeer_v1_{args.T}_{args.alpha}_{args.beta}_{args.player_lambda}_{args.eta}_temp{args.temperature}_p{args.top_p}_initial_log_softmax_t-1_greedy.jsonl"
    args.output_file = os.path.join(args.output_dir, output_filename)
    
    return args

if __name__ == "__main__":
    args = parse_args()
    read_json_and_extract_info(args)