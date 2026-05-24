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
        # Align dimensions
        batch_size, vocab_size = llm_logits.shape
        
        # === Variable Initialization ===
        log_player = torch.log_softmax(llm_logits, dim=-1)      # [batch, vocab]
        log_ref = torch.log_softmax(llm_logits, dim=-1)        # [batch, vocab]
        
        slm_with_logits = torch.log_softmax(slm_with_logits, dim=-1)  # [batch, vocab]
        slm_wo_logits = torch.log_softmax(slm_wo_logits, dim=-1)      # [batch, vocab]
        
        Q = torch.zeros((batch_size, self.iteration_num + 1, vocab_size), 
                        device=llm_logits.device)
        
        log_players_0 = log_player.clone()  # Initial policy memory
        
        log_player_mem = torch.zeros_like(Q)  # Policy memory buffer
        
        # === Iterative Optimization ===
        for cur_iter in range(1, self.iteration_num + 1):
            log_player_mem[:, cur_iter - 1] = log_player.detach()
            Q[:, cur_iter] = self.alpha * (log_player - log_ref) + self.beta * (slm_with_logits - slm_wo_logits)
            
            # Policy update formula
            term1 = cur_iter * self.player_lambda * log_players_0
            term2 = torch.sum(Q[:, :cur_iter + 1], dim=1)
            term3 = log_player_mem[:, cur_iter - 1] / self.eta
            
            denominator = cur_iter * self.player_lambda + 1 / self.eta

            log_player = (term1 + term2 + term3) / denominator
            
            log_player = torch.log_softmax(log_player, dim=-1)
            
        return log_player

def load_models(args):
    """Load models and tokenizers."""
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

# +++++++++++++++++++++++++++++++++++++++++++++++++++++
# +++ Change: Build mapping using vocabulary intersection +++
# +++++++++++++++++++++++++++++++++++++++++++++++++++++
def create_vocab_intersection_map(llm_tokenizer, slm_tokenizer, device):
    """
    Creates the vocabulary intersection of the LLM and SLM, and returns aligned token ID tensors.
    """
    print("Creating vocabulary intersection map...")
    llm_vocab = llm_tokenizer.get_vocab()
    slm_vocab = slm_tokenizer.get_vocab()

    # Find common token strings
    llm_tokens = set(llm_vocab.keys())
    slm_tokens = set(slm_vocab.keys())
    intersect_tokens = llm_tokens.intersection(slm_tokens)

    print(f"LLM vocab size: {len(llm_vocab)}")
    print(f"SLM vocab size: {len(slm_vocab)}")
    print(f"Vocabulary intersection size: {len(intersect_tokens)} tokens.")

    # Create aligned token ID lists for the intersection
    llm_ids_list = []
    slm_ids_list = []
    # Sort to ensure the mapping is deterministic across runs
    for token in sorted(list(intersect_tokens)):
        llm_ids_list.append(llm_vocab[token])
        slm_ids_list.append(slm_vocab[token])

    # Convert to tensors
    llm_intersect_ids = torch.tensor(llm_ids_list, dtype=torch.long, device=device)
    slm_intersect_ids = torch.tensor(slm_ids_list, dtype=torch.long, device=device)
    
    return llm_intersect_ids, slm_intersect_ids

def make_top_5_prompt(query, top_5):
    """Creates a prompt with the top 5 documents as context."""
    prompt_parts = ["The following are five titles with their abstracts."]
    items_to_use = (top_5 or [])[:5]
    for i, item in enumerate(items_to_use):
        prompt_parts.append(f"Title[{i+1}]: {item['title']}\nAbstract[{i+1}]: {item['abstract']}\n")
    prompt_parts.append("Now it's your turn\n")
    prompt_parts.append(query)
    return "\n".join(prompt_parts)

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# +++ Major Change: generate_response function updated to use vocabulary intersection +++
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def generate_response(query_wo, query_with, LLM_model, SLM_model, LLM_tokenizer, SLM_tokenizer, llm_intersect_ids, slm_intersect_ids, args):
    """Handles the generation for a single item."""
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
        # +++ New: Forced Stop Mechanism +++
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # Check if the probability of the original SLM with context outputting eos_token exceeds the threshold.
        # This prevents the Costeer optimization process from diluting a strong stop signal from the SLM with context.
        
        # Convert the original SLM with context logits to a probability distribution
        llm_probs = torch.softmax(llm_logits_native, dim=-1)
        llm_eos_prob = llm_probs[0, LLM_tokenizer.eos_token_id]
        
        # If the probability is greater than the set threshold (e.g., 0.5), force stop the loop.
        # This threshold can be adjusted as needed, or even passed as a new hyperparameter.
        if llm_eos_prob.item() > args.eos_force_threshold:
            print(f"\nINFO: LLM EOS probability ({llm_eos_prob.item():.4f}) exceeded threshold ({args.eos_force_threshold}). Forcing generation to stop.")
            break
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # +++                   End of Changes                     +++
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

        # --- Vocabulary Alignment (Intersection) ---
        intersect_logits_llm = llm_logits_native.index_select(-1, llm_intersect_ids)
        intersect_logits_slm_wo = slm_wo_logits_native.index_select(-1, slm_intersect_ids)
        intersect_logits_slm_with = slm_with_logits_native.index_select(-1, slm_intersect_ids)
        
        # Perform Costeer optimization
        combined_logits = costeer_optimizer.optimize_policy(
            intersect_logits_llm, intersect_logits_slm_wo, intersect_logits_slm_with
        )
        
        # Select the *index* of the next token from the common vocabulary space
        probs = F.softmax(combined_logits, dim=-1)
        next_token_idx = torch.argmax(probs, dim=-1)
        
        # --- Token Reverse Mapping ---
        next_llm_token = llm_intersect_ids[next_token_idx]
        next_slm_token = slm_intersect_ids[next_token_idx]
        
        # Update sequences
        llm_seq = torch.cat((llm_seq, next_llm_token.view(1, 1)), dim=1)
        slm_seq_wo = torch.cat((slm_seq_wo, next_slm_token.view(1, 1)), dim=1)
        slm_seq_with = torch.cat((slm_seq_with, next_slm_token.view(1, 1)), dim=1)

        # Keep the original stopping condition as a fallback.
        if next_llm_token.item() == LLM_tokenizer.eos_token_id:
            break

    generated_text = LLM_tokenizer.decode(
        llm_seq[0, len(llm_inputs.input_ids[0]):], 
        skip_special_tokens=True
    )
    
    return generated_text

def read_json_and_extract_info(args):
    """Reads input, generates responses, and writes to output, with resume capability."""
    LLM_model, SLM_model, LLM_tokenizer, SLM_tokenizer = load_models(args)
    
    # +++ Change: Create intersection map +++
    llm_intersect_ids, slm_intersect_ids = create_vocab_intersection_map(LLM_tokenizer, SLM_tokenizer, LLM_model.device)
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # +++ Change: Use 'input' as a unique ID to resume from a checkpoint +++
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    processed_inputs = set()
    # Check if output file exists, if so, load already processed 'input' content
    if os.path.exists(args.output_file):
        print(f"Found existing output file: {args.output_file}. Loading processed 'input' entries...")
        with open(args.output_file, 'r', encoding='utf-8') as f_out:
            for line in f_out:
                try:
                    processed_item = json.loads(line)
                    # Add the 'input' field content to the set
                    if 'input' in processed_item:
                        processed_inputs.add(processed_item['input'])
                except json.JSONDecodeError:
                    print(f"Warning: A line in the output file could not be parsed and was skipped: {line.strip()}")
        print(f"Loading complete. Found {len(processed_inputs)} processed 'input' entries. Resuming task...")
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # +++                   End of Changes                     +++
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    with open(args.input_file, 'r', encoding='utf-8') as file:
        for idx, line in enumerate(file):
            if args.limit is not None and idx >= args.limit:
                break
            item = json.loads(line)
            item_id = item.get('id')
            user_input = item.get('input')
            if user_input in processed_inputs:
                # For cleaner logs, print only the input prefix
                print(f"Input: '{user_input[:70]}...' already processed, skipping.")
                continue # Skip to the next iteration
            top_5 = item.get('top_5')
            prompt_wo_user_profile = user_input
            prompt_with_user_profile = make_top_5_prompt(user_input, top_5)
            
            response = generate_response(prompt_wo_user_profile, prompt_with_user_profile, 
                                           LLM_model, SLM_model, LLM_tokenizer, SLM_tokenizer,
                                           llm_intersect_ids, slm_intersect_ids, args) # Pass the intersection map
            
            new_json = {
                "id": item_id,
                'input': user_input,
                'response': response,
            }
            with open(args.output_file, 'a', encoding='utf-8') as output_file:
                json.dump(new_json, output_file, ensure_ascii=False)
                output_file.write('\n')

def parse_args():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="Costeer Generator with hyperparameters")
    parser.add_argument("--T", type=int, default=20, help="Number of iterations")
    parser.add_argument("--alpha", type=float, default=2, help="Alpha parameter")
    parser.add_argument("--beta", type=float, default=1, help="Beta parameter")
    parser.add_argument("--player_lambda", type=float, default=2, help="Player lambda parameter")
    parser.add_argument("--eta", type=float, default=10, help="Eta parameter")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p sampling parameter")
    parser.add_argument("--max_new_tokens", type=int, default=1024, help="Maximum number of new tokens to generate")
    parser.add_argument("--input_file", type=str, default="datasets/abstract.jsonl", help="Path to input JSONL file")
    parser.add_argument("--output_dir", type=str, default="outputs/costeer_map", help="Directory for generated output files")
    parser.add_argument("--llm_model_name", type=str, default="Qwen/Qwen2.5-7B-Instruct", help="Path or name of the LLM model")
    parser.add_argument("--slm_model_name", type=str, default="Qwen/Qwen2.5-1.5B-Instruct", help="Path or name of the SLM model")
    parser.add_argument("--eos_force_threshold", type=float, default=0.5, help="Threshold for forcing generation to stop")
    parser.add_argument("--limit", type=int, default=None, help="Optional maximum number of JSONL rows to process")
    
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
