import torch
import torch.nn.functional as F
import json
import argparse
import math

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.logits_process import (
    TemperatureLogitsWarper,
    TopPLogitsWarper,
    RepetitionPenaltyLogitsProcessor
)

class CosteerGenerator:
    def __init__(self, T, alpha, beta, player_lambda, eta, args):
        self.iteration_num = T
        self.alpha = alpha
        self.beta = beta
        self.player_lambda = player_lambda
        self.eta = eta
        self.args = args # 接收完整的args参数

        # 新增：置信度门控的状态
        self.fusion_enabled = True  # 初始时，融合是开启的
        self.confident_streak = 0   # 连续高置信度的计数

    # ----------------------------------------------------
    # §1. 功能函数
    # ----------------------------------------------------

    def _llm_uncertainty(self, logits_llm_step: torch.Tensor) -> float:
        """
        计算LLM的不确定性。当不确定性很低（即高置信度）时，可以跳过Costeer融合。
        这里使用 (1 - max_prob) 作为度量，值越低表示置信度越高。
        输入: logits_llm_step - LLM在当前步的原始logits [1, V_llm]
        输出: 不确定性度量值 (float)
        """
        probs = torch.softmax(logits_llm_step, dim=-1)
        pmax = probs.max(dim=-1).values
        return float((1.0 - pmax).item())

    def optimize_policy(self, llm_logits, slm_wo_logits, slm_with_logits):
        """
        Costeer核心优化策略函数。
        输入logits应为在'交集词汇表'上对齐后的logits。
        """
        # 维度对齐
        batch_size, vocab_size = llm_logits.shape

        # === 变量初始化 ===
        log_player = torch.log_softmax(llm_logits, dim=-1)
        log_ref = log_player.clone() # 初始参考策略就是LLM策略
        
        slm_with_log_probs = torch.log_softmax(slm_with_logits, dim=-1)
        slm_wo_log_probs = torch.log_softmax(slm_wo_logits, dim=-1)

        Q = torch.zeros((batch_size, self.iteration_num + 1, vocab_size), device=llm_logits.device)
        log_players_0 = log_player.clone()
        log_player_mem = torch.zeros_like(Q)

        # 检查是否需要执行融合
        # 如果置信度门控关闭了融合，则迭代次数 T 为 0
        effective_T = self.iteration_num if self.fusion_enabled else 0

        # === 迭代优化 ===
        for cur_iter in range(1, effective_T + 1):
            log_player_mem[:, cur_iter - 1] = log_player.detach()
            
            # 使用固定的 self.beta
            Q[:, cur_iter] = self.alpha * (log_player - log_ref) + self.beta * (slm_with_log_probs - slm_wo_log_probs)
            
            # 分子部分
            term1 = cur_iter * self.player_lambda * log_players_0
            term2 = torch.sum(Q[:, :cur_iter + 1], dim=1)
            term3 = log_player_mem[:, cur_iter - 1] / self.eta
            
            # 分母部分
            denominator = cur_iter * self.player_lambda + 1 / self.eta

            # 更新当前策略
            log_player = (term1 + term2 + term3) / denominator
            log_player = torch.log_softmax(log_player, dim=-1)
        
        # 返回最后一步策略的logits（隐式exp）
        return log_player

# ----------------------------------------------------
# §2. 模型加载与辅助函数 (保持不变或微调)
# ----------------------------------------------------
def create_vocab_intersection_map(llm_tokenizer, slm_tokenizer, device):
    """
    创建LLM和SLM的词汇表交集，并返回用于索引的张量。
    """
    llm_vocab = llm_tokenizer.get_vocab()
    slm_vocab = slm_tokenizer.get_vocab()

    intersect_tokens = set(llm_vocab.keys()).intersection(slm_vocab.keys())

    llm_ids_list = [llm_vocab[token] for token in intersect_tokens]
    slm_ids_list = [slm_vocab[token] for token in intersect_tokens]

    llm_intersect_ids = torch.tensor(llm_ids_list, dtype=torch.long, device=device)
    slm_intersect_ids = torch.tensor(slm_ids_list, dtype=torch.long, device=device)
    
    return llm_intersect_ids, slm_intersect_ids

def load_models(args):
    """加载模型和分词器"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    LLM_model = AutoModelForCausalLM.from_pretrained(
        args.llm_model_name, torch_dtype="auto", device_map="auto"
    ).eval()

    SLM_model = AutoModelForCausalLM.from_pretrained(
        args.slm_model_name, torch_dtype="auto", device_map="auto"
    ).eval()
    
    LLM_tokenizer = AutoTokenizer.from_pretrained(args.llm_model_name)
    SLM_tokenizer = AutoTokenizer.from_pretrained(args.slm_model_name)

    # 新增: 创建词汇表映射
    llm_map, slm_map = create_vocab_intersection_map(LLM_tokenizer, SLM_tokenizer, LLM_model.device)
    
    return LLM_model, SLM_model, LLM_tokenizer, SLM_tokenizer, llm_map, slm_map

def make_top_5_prompt(query, top_5):
    """创建带上下文的prompt (保持不变)"""
    prompt_parts = ["The following are five titles with their abstracts."]
    items_to_use = top_5[:5]
    for i, item in enumerate(items_to_use):
        prompt_parts.append(f"Title[{i+1}]: {item['title']}\nAbstract[{i+1}]: {item['abstract']}\n")
    prompt_parts.append("Now it's your turn\n")
    prompt_parts.append(query)
    return "\n".join(prompt_parts)

# ----------------------------------------------------
# §3. 核心生成逻辑 (重大重构)
# ----------------------------------------------------
def generate_response(query_wo, query_with, LLM_model, SLM_model, LLM_tokenizer, SLM_tokenizer, llm_map, slm_map, args):
    """
    处理单个item的生成，集成了词汇表对齐、采样和动态调度。
    """
    # 准备消息 (保持不变)
    messages_wo = [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": query_wo}]
    messages_with = [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": query_with}]

    # 生成模板 (保持不变)
    llm_text = LLM_tokenizer.apply_chat_template(messages_wo, tokenize=False, add_generation_prompt=True)
    slm_text_wo = SLM_tokenizer.apply_chat_template(messages_wo, tokenize=False, add_generation_prompt=True)
    slm_text_with = SLM_tokenizer.apply_chat_template(messages_with, tokenize=False, add_generation_prompt=True)

    # 准备输入 (保持不变)
    llm_inputs = LLM_tokenizer([llm_text], return_tensors="pt").to(LLM_model.device)
    slm_inputs_wo = SLM_tokenizer([slm_text_wo], return_tensors="pt").to(SLM_model.device)
    slm_inputs_with = SLM_tokenizer([slm_text_with], return_tensors="pt").to(SLM_model.device)
    
    # 初始化KV Caches
    past_key_values_llm = None
    past_key_values_slm_wo = None
    past_key_values_slm_with = None

    # 生成序列初始化
    llm_seq = llm_inputs.input_ids
    slm_seq_wo = slm_inputs_wo.input_ids
    slm_seq_with = slm_inputs_with.input_ids

    # 初始化Costeer优化器，并传入args
    costeer_optimizer = CosteerGenerator(T=args.T, alpha=args.alpha, beta=args.beta, 
                                      player_lambda=args.player_lambda, eta=args.eta, args=args)
    
    # 新增: 初始化采样处理器
    logits_processors = []
    if args.repetition_penalty != 1.0:
        logits_processors.append(RepetitionPenaltyLogitsProcessor(penalty=args.repetition_penalty))
    
    logits_warpers = []
    if args.temperature is not None and args.temperature != 1.0:
        logits_warpers.append(TemperatureLogitsWarper(args.temperature))
    if args.top_p is not None and args.top_p < 1.0:
        logits_warpers.append(TopPLogitsWarper(top_p=args.top_p))

    # 生成循环
    for step in range(args.max_new_tokens):
        # --- 获取各模型logits (带KV缓存优化) ---
        with torch.no_grad():
            llm_outputs = LLM_model(llm_seq, past_key_values=past_key_values_llm, use_cache=True)
            llm_logits = llm_outputs.logits[:, -1, :]
            past_key_values_llm = llm_outputs.past_key_values

            # --- 置信度门控检查 ---
            # 只有在融合开启时才计算SLM的logits
            if costeer_optimizer.fusion_enabled:
                slm_wo_outputs = SLM_model(slm_seq_wo, past_key_values=past_key_values_slm_wo, use_cache=True)
                slm_wo_logits = slm_wo_outputs.logits[:, -1, :]
                past_key_values_slm_wo = slm_wo_outputs.past_key_values

                slm_with_outputs = SLM_model(slm_seq_with, past_key_values=past_key_values_slm_with, use_cache=True)
                slm_with_logits = slm_with_outputs.logits[:, -1, :]
                past_key_values_slm_with = slm_with_outputs.past_key_values

            # 计算LLM不确定性并更新置信度状态
            llm_unc = costeer_optimizer._llm_uncertainty(llm_logits)
            if llm_unc < args.conf_thr:
                costeer_optimizer.confident_streak += 1
            else:
                costeer_optimizer.confident_streak = 0
            
            # 如果连续高置信度达到阈值，则关闭融合
            if costeer_optimizer.fusion_enabled and costeer_optimizer.confident_streak >= args.conf_patience:
                print(f"--- [Step {step}] Confidence gate triggered. Disabling Costeer fusion. ---")
                costeer_optimizer.fusion_enabled = False

        # --- 执行Costeer优化或直接使用LLM logits ---
        if costeer_optimizer.fusion_enabled:
            # 1. 提取交集词汇表的logits
            intersect_llm_logits = llm_logits.index_select(-1, llm_map)
            intersect_slm_wo_logits = slm_wo_logits.index_select(-1, slm_map)
            intersect_slm_with_logits = slm_with_logits.index_select(-1, slm_map)
            
            # 2. 执行Costeer优化
            combined_log_probs = costeer_optimizer.optimize_policy(
                intersect_llm_logits, intersect_slm_wo_logits, intersect_slm_with_logits
            )
            # Costeer返回的是log_probs，采样需要logits，所以直接用它
            final_scores = combined_log_probs
        else:
            # 如果融合关闭，只使用LLM的交集logits
            final_scores = llm_logits.index_select(-1, llm_map)

        # --- 采样 ---
        # 应用RepetitionPenalty等处理器
        for processor in logits_processors:
            final_scores = processor(llm_inputs.input_ids, final_scores) # 注意：这里用llm_inputs.input_ids
        
        # 应用Temperature, Top-p等Warper
        for warper in logits_warpers:
            final_scores = warper(llm_inputs.input_ids, final_scores)
        
        if args.greedy:
            next_token_idx = torch.argmax(final_scores, dim=-1)
        else:
            probs = F.softmax(final_scores, dim=-1)
            next_token_idx = torch.multinomial(probs, num_samples=1).squeeze(-1)
        
        # --- 更新序列 ---
        # 将交集词汇表的索引映射回各自模型的token id
        next_token_llm = llm_map[next_token_idx]
        next_token_slm = slm_map[next_token_idx]

        # 更新输入序列以进行下一步生成
        llm_seq = next_token_llm.unsqueeze(0)
        if costeer_optimizer.fusion_enabled:
            slm_seq_wo = next_token_slm.unsqueeze(0)
            slm_seq_with = next_token_slm.unsqueeze(0)
        
        # 记录生成的token (使用LLM的ID)
        llm_inputs.input_ids = torch.cat([llm_inputs.input_ids, next_token_llm.view(1, 1)], dim=-1)
        
        if next_token_llm.item() == LLM_tokenizer.eos_token_id:
            break

    # 解码生成文本
    generated_text = LLM_tokenizer.decode(
        llm_inputs.input_ids[0, len(llm_inputs.input_ids[0]) - step -1 :], 
        skip_special_tokens=True
    )
    
    return generated_text

# ----------------------------------------------------
# §4. 主程序流程 (修改以传递新参数)
# ----------------------------------------------------

def read_json_and_extract_info(args):
    LLM_model, SLM_model, LLM_tokenizer, SLM_tokenizer, llm_map, slm_map = load_models(args)
    
    with open(args.input_file, 'r', encoding='utf-8') as file:
        for line in file:
            item = json.loads(line)
            id = item.get('id')
            input_query = item.get('input')
            top_5 = item.get('top_5')
            
            prompt_wo_user_profile = input_query
            prompt_with_user_profile = make_top_5_prompt(input_query, top_5)
            
            response = generate_response(
                prompt_wo_user_profile, prompt_with_user_profile, 
                LLM_model, SLM_model, LLM_tokenizer, SLM_tokenizer, 
                llm_map, slm_map, args
            )
            
            new_json = {"id": id, 'input': input_query, 'response': response}
            with open(args.output_file, 'a', encoding='utf-8') as output_file:
                json.dump(new_json, output_file, ensure_ascii=False)
                output_file.write('\n')

def parse_args():
    parser = argparse.ArgumentParser(description="Costeer Generator with Dynamic Scheduling and Sampling")
    # --- 原有Costeer参数 ---
    parser.add_argument("--T", type=int, default=20, help="Number of iterations")
    parser.add_argument("--alpha", type=float, default=2, help="Alpha parameter")
    parser.add_argument("--beta", type=float, default=1, help="Beta parameter (fixed)")
    parser.add_argument("--player_lambda", type=float, default=2, help="Player lambda parameter")
    parser.add_argument("--eta", type=float, default=10, help="Eta parameter")
    
    # --- 采样参数 ---
    parser.add_argument("--greedy", action='store_true', help="Use greedy decoding instead of sampling.")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p (nucleus) sampling parameter")
    parser.add_argument("--repetition_penalty", type=float, default=1.0, help="Repetition penalty")

    # --- 置信度门控参数 ---
    parser.add_argument("--conf_thr", type=float, default=0.1, help="Confidence threshold for gating. Lower means more confident.")
    parser.add_argument("--conf_patience", type=int, default=3, help="Num consecutive confident tokens to disable Costeer fusion.")

    # --- IO和模型路径 ---
    parser.add_argument("--max_new_tokens", type=int, default=1024, help="Maximum number of new tokens to generate")
    parser.add_argument("--input_file", type=str, default="datasets/longlamp/abstract.jsonl", help="Path to input JSONL file")
    parser.add_argument("--output_dir", type=str, default="rebuttal/results/scheduler", help="Directory for output files")
    parser.add_argument("--llm_model_name", type=str, default="models/Qwen2.5-7B-Instruct", help="Path to LLM model")
    parser.add_argument("--slm_model_name", type=str, default="models/Qwen2.5-1.5B-Instruct", help="Path to SLM model")
    
    args = parser.parse_args()
    
    # --- 动态生成输出文件名 (更新) ---
    llm_model_short = args.llm_model_name.split('/')[-1]
    slm_model_short = args.slm_model_name.split('/')[-1]
    
    sampling_mode = "greedy" if args.greedy else f"temp{args.temperature}_p{args.top_p}"
    output_filename = (
        f"abstract_{llm_model_short}_{slm_model_short}_"
        f"thr{args.conf_thr}_patience{args.conf_patience}_"
        f"T{args.T}_alpha{args.alpha}_beta{args.beta}_lambda{args.player_lambda}_"
        f"{sampling_mode}_gated.jsonl"
    )
    args.output_file = f"{args.output_dir}/{output_filename}"
    
    return args

if __name__ == "__main__":
    args = parser_args()
    read_json_and_extract_info(args)