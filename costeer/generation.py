"""Shared-vocabulary, per-token CoSteer generation with KV caching."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from .optimizer import CoSteerOptimizer


@dataclass
class ModelBundle:
    llm: Any
    slm: Any
    llm_tokenizer: Any
    slm_tokenizer: Any


def load_model_bundle(
    llm_name: str,
    slm_name: str,
    llm_device_map: str = "auto",
    slm_device_map: str = "auto",
    trust_remote_code: bool = False,
) -> ModelBundle:
    common = {"torch_dtype": "auto", "trust_remote_code": trust_remote_code}
    llm = AutoModelForCausalLM.from_pretrained(
        llm_name, device_map=llm_device_map, **common
    )
    slm = AutoModelForCausalLM.from_pretrained(
        slm_name, device_map=slm_device_map, **common
    )
    llm.eval()
    slm.eval()
    llm_tokenizer = AutoTokenizer.from_pretrained(
        llm_name, trust_remote_code=trust_remote_code
    )
    slm_tokenizer = AutoTokenizer.from_pretrained(
        slm_name, trust_remote_code=trust_remote_code
    )
    validate_shared_vocabulary(llm_tokenizer, slm_tokenizer)
    return ModelBundle(llm, slm, llm_tokenizer, slm_tokenizer)


def validate_shared_vocabulary(llm_tokenizer: Any, slm_tokenizer: Any) -> None:
    if len(llm_tokenizer) != len(slm_tokenizer):
        raise ValueError(
            "The canonical runner requires identical LLM/SLM token IDs. "
            "Use codes/abstract_costeer_map.py or codes/abstract_costeer_byte.py "
            "for tokenizer-mismatch experiments."
        )
    if llm_tokenizer.get_vocab() != slm_tokenizer.get_vocab():
        raise ValueError("LLM and SLM tokenizers have different token-to-id mappings")


def _model_input_device(model: Any) -> torch.device:
    return model.get_input_embeddings().weight.device


def _chat_text(tokenizer: Any, prompt: str) -> str:
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt},
    ]
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )


def _next_token(
    log_policy: torch.Tensor,
    do_sample: bool,
    temperature: float,
    top_p: float,
) -> torch.Tensor:
    if not do_sample:
        return torch.argmax(log_policy, dim=-1, keepdim=True)
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if not 0 < top_p <= 1:
        raise ValueError("top_p must be in (0, 1]")

    logits = log_policy / temperature
    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        sorted_probs = F.softmax(sorted_logits, dim=-1)
        remove = torch.cumsum(sorted_probs, dim=-1) > top_p
        remove[..., 1:] = remove[..., :-1].clone()
        remove[..., 0] = False
        sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
        logits = torch.full_like(logits, float("-inf")).scatter(
            -1, sorted_indices, sorted_logits
        )
    return torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)


@torch.inference_mode()
def generate_with_costeer(
    input_text: str,
    personalized_input_text: str,
    models: ModelBundle,
    optimizer: CoSteerOptimizer,
    max_new_tokens: int = 1024,
    do_sample: bool = False,
    temperature: float = 1.0,
    top_p: float = 0.9,
) -> str:
    llm_device = _model_input_device(models.llm)
    slm_device = _model_input_device(models.slm)

    llm_prompt = _chat_text(models.llm_tokenizer, input_text)
    slm_prompt_without = _chat_text(models.slm_tokenizer, input_text)
    slm_prompt_with = _chat_text(models.slm_tokenizer, personalized_input_text)

    llm_ids = models.llm_tokenizer(
        [llm_prompt], return_tensors="pt"
    ).input_ids.to(llm_device)
    slm_ids_without = models.slm_tokenizer(
        [slm_prompt_without], return_tensors="pt"
    ).input_ids.to(slm_device)
    slm_ids_with = models.slm_tokenizer(
        [slm_prompt_with], return_tensors="pt"
    ).input_ids.to(slm_device)

    llm_cache = None
    slm_cache_without = None
    slm_cache_with = None
    generated_ids: list[int] = []
    eos_token_ids = models.llm_tokenizer.eos_token_id
    if isinstance(eos_token_ids, int):
        eos_token_ids = {eos_token_ids}
    else:
        eos_token_ids = set(eos_token_ids or [])

    for _ in range(max_new_tokens):
        llm_output = models.llm(
            input_ids=llm_ids, past_key_values=llm_cache, use_cache=True
        )
        slm_output_without = models.slm(
            input_ids=slm_ids_without,
            past_key_values=slm_cache_without,
            use_cache=True,
        )
        slm_output_with = models.slm(
            input_ids=slm_ids_with,
            past_key_values=slm_cache_with,
            use_cache=True,
        )

        llm_cache = llm_output.past_key_values
        slm_cache_without = slm_output_without.past_key_values
        slm_cache_with = slm_output_with.past_key_values

        llm_logits = llm_output.logits[:, -1, :]
        slm_logits_without = slm_output_without.logits[:, -1, :].to(llm_logits.device)
        slm_logits_with = slm_output_with.logits[:, -1, :].to(llm_logits.device)
        log_policy = optimizer.optimize_policy(
            llm_logits, slm_logits_without, slm_logits_with
        )
        next_token = _next_token(log_policy, do_sample, temperature, top_p)
        token_id = int(next_token.item())
        generated_ids.append(token_id)

        if token_id in eos_token_ids:
            break
        llm_ids = next_token.to(llm_device)
        slm_ids_without = next_token.to(slm_device)
        slm_ids_with = next_token.to(slm_device)

    return models.llm_tokenizer.decode(generated_ids, skip_special_tokens=True)
