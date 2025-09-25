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
        self.args = args # Store the full arguments

        # New: State for confidence gating
        self.fusion_enabled = True  # Initially, fusion is enabled
        self.confident_streak = 0   # Counter for consecutive high-confidence steps

    # ----------------------------------------------------
    # §1. Utility Functions
    # ----------------------------------------------------

    def _llm_uncertainty(self, logits_llm_step: torch.Tensor) -> float:
        """
        Calculate the LLM's uncertainty. When uncertainty is low (i.e., high confidence),
        Costeer fusion can be skipped. Here, (1 - max_prob) is used as the metric,
        where a lower value indicates higher confidence.
        Input: logits_llm_step - The LLM's raw logits at the current step [1, V_llm]
        Output: Uncertainty metric (float)
        """
        probs = torch.softmax(logits_llm_step, dim=-1)
        pmax = probs.max(dim=-1).values
        return float((1.0 - pmax).item())

    def optimize_policy(self, llm_logits, slm_wo_logits, slm_with_logits):
        """
        The core optimization policy function for Costeer.
        Input logits should be aligned on the 'vocabulary intersection'.
        """
        # Align dimensions
        batch_size, vocab_size = llm_logits.shape

        # === Variable Initialization ===
        log_player = torch.log_softmax(llm_logits, dim=-1)
        log_ref = log_player.clone() # The initial reference policy is the LLM's policy

        slm_with_log_probs = torch.log_softmax(slm_with_logits, dim=-1)
        slm_wo_log_probs = torch.log_softmax(slm_wo_logits, dim=-1)

        Q = torch.zeros((batch_size, self.iteration_num + 1, vocab_size), device=llm_logits.device)
        log_players_0 = log_player.clone()
        log_player_mem = torch.zeros_like(Q)

        # Check if fusion needs to be performed
        # If confidence gating has disabled fusion, the number of iterations T is 0
        effective_T = self.iteration_num if self.fusion_enabled else 0

        # === Iterative Optimization ===
        for cur_iter in range(1, effective_T + 1):
            log_player_mem[:, cur_iter - 1] = log_player.detach()

            # Use the fixed self.beta
            Q[:, cur_iter] = self.alpha * (log_player - log_ref) + self.beta * (slm_with_log_probs - slm_wo_log_probs)

            # Numerator part
            term1 = cur_iter * self.player_lambda * log_players_0
            term2 = torch.sum(Q[:, :cur_iter + 1], dim=1)
            term3 = log_player_mem[:, cur_iter - 1] / self.eta

            # Denominator part
            denominator = cur_iter * self.player_lambda + 1 / self.eta

            # Update the current policy
            log_player = (term1 + term2 + term3) / denominator
            log_player = torch.log_softmax(log_player, dim=-1)

        # Return the logits of the last step's policy (implicit exp)
        return log_player

# ----------------------------------------------------
# §2. Model Loading and Helper Functions
# ----------------------------------------------------
def create_vocab_intersection_map(llm_tokenizer, slm_tokenizer, device):
    """
    Create the vocabulary intersection of the LLM and SLM, and return tensors for indexing.
    """
    llm_vocab = llm_tokenizer.get_vocab()
    slm_vocab = slm_tokenizer.get_vocab()

    intersect_tokens = set(llm_vocab.keys()).intersection(slm_vocab.keys())

    llm_ids_list = [llm_vocab[token] for token in sorted(list(intersect_tokens))]
    slm_ids_list = [slm_vocab[token] for token in sorted(list(intersect_tokens))]

    llm_intersect_ids = torch.tensor(llm_ids_list, dtype=torch.long, device=device)
    slm_intersect_ids = torch.tensor(slm_ids_list, dtype=torch.long, device=device)

    return llm_intersect_ids, slm_intersect_ids

def load_models(args):
    """Load models and tokenizers"""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    LLM_model = AutoModelForCausalLM.from_pretrained(
        args.llm_model_name, torch_dtype="auto", device_map="auto"
    ).eval()

    SLM_model = AutoModelForCausalLM.from_pretrained(
        args.slm_model_name, torch_dtype="auto", device_map="auto"
    ).eval()

    LLM_tokenizer = AutoTokenizer.from_pretrained(args.llm_model_name)
    SLM_tokenizer = AutoTokenizer.from_pretrained(args.slm_model_name)

    # New: Create vocabulary mapping
    llm_map, slm_map = create_vocab_intersection_map(LLM_tokenizer, SLM_tokenizer, LLM_model.device)

    return LLM_model, SLM_model, LLM_tokenizer, SLM_tokenizer, llm_map, slm_map

def make_top_5_prompt(query, top_5):
    """Create a prompt with context"""
    prompt_parts = ["The following are five titles with their abstracts."]
    items_to_use = top_5[:5]
    for i, item in enumerate(items_to_use):
        prompt_parts.append(f"Title[{i+1}]: {item['title']}\nAbstract[{i+1}]: {item['abstract']}\n")
    prompt_parts.append("Now it's your turn\n")
    prompt_parts.append(query)
    return "\n".join(prompt_parts)

# ----------------------------------------------------
# §3. Core Generation Logic
# ----------------------------------------------------
def generate_response(query_wo, query_with, LLM_model, SLM_model, LLM_tokenizer, SLM_tokenizer, llm_map, slm_map, args):
    """
    Handle the generation for a single item, integrating vocabulary alignment,
    sampling, and dynamic scheduling.
    """
    # Prepare messages
    messages_wo = [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": query_wo}]
    messages_with = [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": query_with}]

    # Generate template
    llm_text = LLM_tokenizer.apply_chat_template(messages_wo, tokenize=False, add_generation_prompt=True)
    slm_text_wo = SLM_tokenizer.apply_chat_template(messages_wo, tokenize=False, add_generation_prompt=True)
    slm_text_with = SLM_tokenizer.apply_chat_template(messages_with, tokenize=False, add_generation_prompt=True)

    # Prepare inputs
    llm_inputs = LLM_tokenizer([llm_text], return_tensors="pt").to(LLM_model.device)
    slm_inputs_wo = SLM_tokenizer([slm_text_wo], return_tensors="pt").to(SLM_model.device)
    slm_inputs_with = SLM_tokenizer([slm_text_with], return_tensors="pt").to(SLM_model.device)

    # Initialize KV Caches
    past_key_values_llm = None
    past_key_values_slm_wo = None
    past_key_values_slm_with = None

    # Initialize generation sequences
    llm_seq = llm_inputs.input_ids
    slm_seq_wo = slm_inputs_wo.input_ids
    slm_seq_with = slm_inputs_with.input_ids

    # Initialize the Costeer optimizer and pass args
    costeer_optimizer = CosteerGenerator(T=args.T, alpha=args.alpha, beta=args.beta,
                                       player_lambda=args.player_lambda, eta=args.eta, args=args)

    # New: Initialize sampling processors
    logits_processors = []
    if args.repetition_penalty != 1.0:
        logits_processors.append(RepetitionPenaltyLogitsProcessor(penalty=args.repetition_penalty))

    logits_warpers = []
    if args.temperature is not None and args.temperature != 1.0:
        logits_warpers.append(TemperatureLogitsWarper(args.temperature))
    if args.top_p is not None and args.top_p < 1.0:
        logits_warpers.append(TopPLogitsWarper(top_p=args.top_p))

    # Generation loop
    for step in range(args.max_new_tokens):
        # --- Get logits from each model (with KV cache optimization) ---
        with torch.no_grad():
            llm_outputs = LLM_model(llm_seq, past_key_values=past_key_values_llm, use_cache=True)
            llm_logits = llm_outputs.logits[:, -1, :]
            past_key_values_llm = llm_outputs.past_key_values

            # --- Confidence Gating Check ---
            # Only compute SLM logits if fusion is enabled
            if costeer_optimizer.fusion_enabled:
                slm_wo_outputs = SLM_model(slm_seq_wo, past_key_values=past_key_values_slm_wo, use_cache=True)
                slm_wo_logits = slm_wo_outputs.logits[:, -1, :]
                past_key_values_slm_wo = slm_wo_outputs.past_key_values

                slm_with_outputs = SLM_model(slm_seq_with, past_key_values=past_key_values_slm_with, use_cache=True)
                slm_with_logits = slm_with_outputs.logits[:, -1, :]
                past_key_values_slm_with = slm_with_outputs.past_key_values

            # Calculate LLM uncertainty and update confidence state
            llm_unc = costeer_optimizer._llm_uncertainty(llm_logits)
            if llm_unc < args.conf_thr:
                costeer_optimizer.confident_streak += 1
            else:
                costeer_optimizer.confident_streak = 0

            # If the consecutive high-confidence streak reaches the threshold, disable fusion
            if costeer_optimizer.fusion_enabled and costeer_optimizer.confident_streak >= args.conf_patience:
                print(f"--- [Step {step}] Confidence gate triggered. Disabling Costeer fusion. ---")
                costeer_optimizer.fusion_enabled = False

        # --- Perform Costeer optimization or use LLM logits directly ---
        if costeer_optimizer.fusion_enabled:
            # 1. Extract logits for the vocabulary intersection
            intersect_llm_logits = llm_logits.index_select(-1, llm_map)
            intersect_slm_wo_logits = slm_wo_logits.index_select(-1, slm_map)
            intersect_slm_with_logits = slm_with_logits.index_select(-1, slm_map)

            # 2. Perform Costeer optimization
            combined_log_probs = costeer_optimizer.optimize_policy(
                intersect_llm_logits, intersect_slm_wo_logits, intersect_slm_with_logits
            )
            # Costeer returns log_probs; we can use them directly as scores for sampling
            final_scores = combined_log_probs
        else:
            # If fusion is disabled, only use the LLM's intersection logits
            final_scores = torch.log_softmax(llm_logits.index_select(-1, llm_map), dim=-1)


        # --- Sampling ---
        # Apply processors like RepetitionPenalty
        for processor in logits_processors:
            final_scores = processor(llm_inputs.input_ids, final_scores) # Note: Using llm_inputs.input_ids here

        # Apply warpers like Temperature, Top-p
        for warper in logits_warpers:
            final_scores = warper(llm_inputs.input_ids, final_scores)

        if args.greedy:
            next_token_idx = torch.argmax(final_scores, dim=-1)
        else:
            probs = F.softmax(final_scores, dim=-1)
            next_token_idx = torch.multinomial(probs, num_samples=1).squeeze(-1)

        # --- Update Sequences ---
        # Map the index from the intersection vocabulary back to the respective model's token id
        next_token_llm = llm_map[next_token_idx]
        next_token_slm = slm_map[next_token_idx]

        # Update the input sequence for the next generation step
        llm_seq = next_token_llm.unsqueeze(0)
        if costeer_optimizer.fusion_enabled:
            slm_seq_wo = next_token_slm.unsqueeze(0)
            slm_seq_with = next_token_slm.unsqueeze(0)

        # Record the generated token (using the LLM's ID)
        llm_inputs.input_ids = torch.cat([llm_inputs.input_ids, next_token_llm.view(1, 1)], dim=-1)

        if next_token_llm.item() == LLM_tokenizer.eos_token_id:
            break

    # Decode the generated text
    generated_text = LLM_tokenizer.decode(
        llm_inputs.input_ids[0, len(llm_inputs.input_ids[0]) - step - 1:],
        skip_special_tokens=True
    )

    return generated_text

# ----------------------------------------------------
# §4. Main Program Flow
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
    # --- Original Costeer Parameters ---
    parser.add_argument("--T", type=int, default=20, help="Number of iterations")
    parser.add_argument("--alpha", type=float, default=2, help="Alpha parameter")
    parser.add_argument("--beta", type=float, default=1, help="Beta parameter (fixed)")
    parser.add_argument("--player_lambda", type=float, default=2, help="Player lambda parameter")
    parser.add_argument("--eta", type=float, default=10, help="Eta parameter")

    # --- Sampling Parameters ---
    parser.add_argument("--greedy", action='store_true', help="Use greedy decoding instead of sampling.")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=1.0, help="Top-p (nucleus) sampling parameter")
    parser.add_argument("--repetition_penalty", type=float, default=1.0, help="Repetition penalty")

    # --- Confidence Gating Parameters ---
    parser.add_argument("--conf_thr", type=float, default=0.1, help="Confidence threshold for gating. Lower means more confident.")
    parser.add_argument("--conf_patience", type=int, default=3, help="Num consecutive confident tokens to disable Costeer fusion.")

    # --- I/O and Model Paths ---
    parser.add_argument("--max_new_tokens", type=int, default=1024, help="Maximum number of new tokens to generate")
    parser.add_argument("--input_file", type=str, default="datasets/longlamp/abstract.jsonl", help="Path to input JSONL file")
    parser.add_argument("--output_dir", type=str, default="results/adacosteer", help="Directory for output files")
    parser.add_argument("--llm_model_name", type=str, default="models/Qwen2.5-7B-Instruct", help="Path to LLM model")
    parser.add_argument("--slm_model_name", type=str, default="models/Qwen2.5-1.5B-Instruct", help="Path to SLM model")

    args = parser.parse_args()

    # --- Dynamically Generate Output Filename (Updated) ---
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
    args = parse_args()
    read_json_and_extract_info(args)
