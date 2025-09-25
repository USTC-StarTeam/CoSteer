from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import json
import torch.nn.functional as F
import argparse

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
        # print(llm_logits.shape, slm_wo_logits.shape, slm_with_logits.shape)
        # === 变量初始化 ===
        
        log_player = llm_logits
        log_player = torch.log(F.softmax(llm_logits, dim=-1))  # [batch, vocab]
  
        log_ref = llm_logits
        log_ref = torch.log(F.softmax(llm_logits, dim=-1))  # [batch, vocab]
        
        slm_with_logits = torch.log(F.softmax(slm_with_logits, dim=-1))  # [batch, vocab]
        slm_wo_logits = torch.log(F.softmax(slm_wo_logits, dim=-1))  # [batch, vocab]
       
        Q = torch.zeros((batch_size, self.iteration_num + 1, vocab_size), 
                    device=llm_logits.device)
        
        
        log_players_0 = log_player.clone()  # 初始策略记忆体
        
        
        log_player_mem = torch.zeros_like(Q)  # 策略记忆体
        # === 迭代优化 ===
        for cur_iter in range(1, self.iteration_num+1):
            log_player_mem[:, cur_iter-1] = log_player.detach()
            
            Q[:, cur_iter] = self.alpha * (log_player - log_ref) + self.beta * (slm_with_logits - slm_wo_logits)  # [batch, vocab]
            
            
            # 分子部分
            term1 = (cur_iter) * self.player_lambda * log_players_0                    # λ * π_1
            term2 = torch.sum(Q[:, 0:cur_iter+1], dim=1)       # (1/t)ΣQ
            term3 = log_player_mem[:, cur_iter-1] / (self.eta) # (1/ηt)π_mem
            
            # 分母部分
            denominator = cur_iter * self.player_lambda + 1/(self.eta)

            # 更新当前策略
            log_player = (term1 + term2 + term3) / denominator
            
            
            log_player = torch.log_softmax(log_player, dim=-1)  # 严格对应源码第21行
            
            
            
        
        
        return log_player

# 加载模型
def load_models(args):
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

def make_top_5_prompt(query, top_5):
    prompt_parts = ["The following are five titles with their abstracts."]
    
    # Only use up to 5 items
    items_to_use = top_5[:5]
    
    for i, item in enumerate(items_to_use):
        prompt_parts.append(f"Title[{i+1}]: {item['title']}\nAbstract[{i+1}]: {item['abstract']}\n")
    
    prompt_parts.append("Now it's your turn\n")
    prompt_parts.append(query)
    
    return "\n".join(prompt_parts)

def generate_response(query_wo, query_with, LLM_model, SLM_model, LLM_tokenizer, SLM_tokenizer, args):
    """处理单个item的生成"""
    # 准备消息
    messages_wo = [{"role": "system", "content": "You are a helpful assistant."},
                 {"role": "user", "content": query_wo}]
    messages_with = [{"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": query_with}]

    # 生成模板
    llm_text_wo = LLM_tokenizer.apply_chat_template(
        messages_wo, tokenize=False, add_generation_prompt=True)
    slm_text_wo = SLM_tokenizer.apply_chat_template(
        messages_wo, tokenize=False, add_generation_prompt=True)
    slm_text_with = SLM_tokenizer.apply_chat_template(
        messages_with, tokenize=False, add_generation_prompt=True)

    # 准备输入
    llm_inputs = LLM_tokenizer([llm_text_wo], return_tensors="pt").to(LLM_model.device)
    slm_inputs_wo = SLM_tokenizer([slm_text_wo], return_tensors="pt").to(SLM_model.device)
    slm_inputs_with = SLM_tokenizer([slm_text_with], return_tensors="pt").to(SLM_model.device)

    # 生成参数
    max_new_tokens = args.max_new_tokens

    # 生成序列初始化
    llm_seq = llm_inputs.input_ids
    slm_seq_wo = slm_inputs_wo.input_ids
    slm_seq_with = slm_inputs_with.input_ids

    # 初始化Costeer优化器
    costeer_optimizer = CosteerGenerator(T=args.T, alpha=args.alpha, beta=args.beta, 
                                      player_lambda=args.player_lambda, eta=args.eta)
    
    # 生成循环
    for _ in range(max_new_tokens):
        # 获取各模型logits（带KV缓存优化）
        with torch.no_grad():
            slm_wo_logits = SLM_model(slm_seq_wo).logits[:, -1, :]
            slm_with_logits = SLM_model(slm_seq_with).logits[:, -1, :]
            llm_logits = LLM_model(llm_seq).logits[:, -1, :][:, :SLM_model(slm_seq_with).logits.size(-1)]
        
        # 执行Costeer优化
        combined_logits = costeer_optimizer.optimize_policy(
            llm_logits, slm_wo_logits, slm_with_logits
        )
        
        # 直接使用softmax获取概率分布
        probs = F.softmax(combined_logits, dim=-1)
        
        # 贪婪解码 - 选择概率最高的token
        next_token = torch.argmax(probs, dim=-1)
        
        # 更新序列
        llm_seq = torch.cat((llm_seq, next_token.view(1, 1)), dim=1)
        slm_seq_wo = torch.cat((slm_seq_wo, next_token.view(1, 1)), dim=1)
        slm_seq_with = torch.cat((slm_seq_with, next_token.view(1, 1)), dim=1)

        if next_token.item() == LLM_tokenizer.eos_token_id:
            break

    # 解码生成文本
    generated_text = LLM_tokenizer.decode(
        llm_seq[0, len(llm_inputs.input_ids[0]):], 
        skip_special_tokens=True
    )
    
    return generated_text

def read_json_and_extract_info(args):
    LLM_model, SLM_model, LLM_tokenizer, SLM_tokenizer = load_models(args)
    
    with open(args.input_file, 'r', encoding='utf-8') as file:
        for line in file:
            #start_time = time.time()
            item = json.loads(line)
            id = item.get('id')
            input = item.get('input')
            top_5 = item.get('top_5')
            prompt_wo_user_profile = input
            prompt_with_user_profile = make_top_5_prompt(input, top_5)
            response = generate_response(prompt_wo_user_profile, prompt_with_user_profile, 
                                        LLM_model, SLM_model, LLM_tokenizer, SLM_tokenizer, args)
            new_json = {
                "id": id,
                'input': input,
                'response': response,
            }
            with open(args.output_file, 'a', encoding='utf-8') as output_file:
                json.dump(new_json, output_file, ensure_ascii=False)
                output_file.write('\n')
            #end_time = time.time()
            #print(f"Processed entry {id} in {end_time - start_time:.2f} seconds")

def parse_args():
    parser = argparse.ArgumentParser(description="Costeer Generator with hyperparameters")
    parser.add_argument("--T", type=int, default=20, help="Number of iterations")
    parser.add_argument("--alpha", type=float, default=2, help="Alpha parameter")
    parser.add_argument("--beta", type=float, default=1, help="Beta parameter")
    parser.add_argument("--player_lambda", type=float, default=2, help="Player lambda parameter")
    parser.add_argument("--eta", type=float, default=10, help="Eta parameter")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p sampling parameter")
    parser.add_argument("--max_new_tokens", type=int, default=4096, help="Maximum number of new tokens to generate")
    parser.add_argument("--input_file", type=str, default="datasets/longlamp/abstract.jsonl", help="Path to input JSONL file")
    parser.add_argument("--output_dir", type=str, default="results", help="Directory for output files")
    parser.add_argument("--llm_model_name", type=str, default="Qwen3-8B", help="Path to LLM model")
    parser.add_argument("--slm_model_name", type=str, default="Qwen3-0.6B", help="Path to SLM model")
    
    args = parser.parse_args()
    
    # Create output filename based on parameters
    llm_model_short = args.llm_model_name.split('/')[-1]
    slm_model_short = args.slm_model_name.split('/')[-1]
    
    output_filename = f"abstract_bgereranker_{llm_model_short}_{slm_model_short}_costeer_v1_{args.T}_{args.alpha}_{args.beta}_{args.player_lambda}_{args.eta}_temp{args.temperature}_p{args.top_p}_initial_log_softmax_t-1_greedy.jsonl"
    args.output_file = f"{args.output_dir}/{output_filename}"
    
    return args

if __name__ == "__main__":
    args = parse_args()
    read_json_and_extract_info(args)