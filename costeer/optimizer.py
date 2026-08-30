"""The per-token CoSteer policy update."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class CoSteerConfig:
    """Hyperparameters used by the CoSteer update in the paper."""

    iterations: int = 20
    alpha: float = 2.0
    beta: float = 1.0
    player_lambda: float = 2.0
    eta: float = 10.0

    def __post_init__(self) -> None:
        if self.iterations < 1:
            raise ValueError("iterations must be at least 1")
        if self.player_lambda <= 0:
            raise ValueError("player_lambda must be positive")
        if self.eta <= 0:
            raise ValueError("eta must be positive")


class CoSteerOptimizer:
    """Fuse a local context delta into the cloud model policy.

    The three inputs are next-token logits over the same token IDs. The local
    delta is the difference between the SLM log distributions with and without
    personal context. FTRL then updates the LLM policy for the current token.
    """

    def __init__(self, config: CoSteerConfig | None = None) -> None:
        self.config = config or CoSteerConfig()

    def optimize_policy(
        self,
        llm_logits: torch.Tensor,
        slm_without_context_logits: torch.Tensor,
        slm_with_context_logits: torch.Tensor,
    ) -> torch.Tensor:
        tensors = (
            llm_logits,
            slm_without_context_logits,
            slm_with_context_logits,
        )
        if any(tensor.ndim != 2 for tensor in tensors):
            raise ValueError("all logits must have shape [batch, vocabulary]")
        if len({tuple(tensor.shape) for tensor in tensors}) != 1:
            raise ValueError("LLM and SLM logits must use the same vocabulary")

        # The optimizer is intentionally evaluated in fp32 for stable repeated
        # log-softmax updates even when model inference uses bf16/fp16.
        base_policy = F.log_softmax(llm_logits.float(), dim=-1)
        local_delta = F.log_softmax(
            slm_with_context_logits.float(), dim=-1
        ) - F.log_softmax(slm_without_context_logits.float(), dim=-1)

        player = base_policy.clone()
        cumulative_utility = torch.zeros_like(player)
        cfg = self.config

        for step in range(1, cfg.iterations + 1):
            utility = cfg.alpha * (player - base_policy) + cfg.beta * local_delta
            cumulative_utility = cumulative_utility + utility
            player = (
                step * cfg.player_lambda * base_policy
                + cumulative_utility
                + player / cfg.eta
            ) / (step * cfg.player_lambda + 1.0 / cfg.eta)
            player = F.log_softmax(player, dim=-1)

        return player
