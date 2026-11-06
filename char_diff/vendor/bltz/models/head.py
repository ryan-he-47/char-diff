"""Conditional byte decoder heads (docs/01-design.md §5.4, D1; FiLM variant 2026-09-14).

(h, delta) -> 256-way byte logits. delta is ALWAYS a relative offset (1..k_max),
never an absolute position (simple_point_cloud v6 static-fan lesson: with
absolute positions the decoder learns to ignore the query). Given one h, all
delta queries are independent -> one batched forward covers the whole horizon.

Two conditioning styles (config model.head_type):
- "concat" (ConditionalByteHead): delta_emb concatenated at the MLP input —
  the delta signal must be carried through 4 layers by the network itself.
- "film" (FiLMByteHead): delta modulates the hidden layers multiplicatively
  (FiLM: LN -> Linear -> gamma⊙x+beta, gamma=1+g_raw/beta=b_raw with a
  zero-initialized conditioner so every block starts EXACTLY at identity).
  delta acts as a controller of how h is read, not as content — a shorter,
  more direct conditioning path, and it matches simple_point_cloud's FiLM
  experiments. Stability infra: RMSNorm before each modulated layer (keeps
  the modulated activations unit-scale under fp16), zero-init conditioner.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import RMSNorm


class ConditionalByteHead(nn.Module):
    def __init__(
        self,
        d_in: int = 768,
        k_max: int = 48,
        d_delta: int = 64,
        hidden: int = 1536,
        depth: int = 4,
    ):
        super().__init__()
        self.delta_emb = nn.Embedding(k_max, d_delta)
        layers: list[nn.Module] = [nn.Linear(d_in + d_delta, hidden), nn.SiLU()]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden, hidden), nn.SiLU()]
        self.mlp = nn.Sequential(*layers)
        self.out = nn.Linear(hidden, 256)

    def forward(self, h: torch.Tensor, delta_idx: torch.Tensor) -> torch.Tensor:
        """h (N, d_in), delta_idx (N,) long in [0, k_max) (i.e. delta - 1) -> (N, 256)."""
        x = torch.cat([h, self.delta_emb(delta_idx)], dim=-1)
        return self.out(self.mlp(x))


class FiLMBlock(nn.Module):
    """LN -> Linear -> FiLM(gamma, beta from delta) -> SiLU.

    gamma = 1 + g_raw, beta = b_raw; the conditioner projection is
    zero-initialized so at init the block is EXACTLY SiLU(Linear(LN(x)))
    for every delta (identity start = no FiLM-induced training shock).
    """

    def __init__(self, hidden: int, d_delta: int):
        super().__init__()
        self.norm = RMSNorm(hidden)
        self.fc = nn.Linear(hidden, hidden)
        self.mod = nn.Linear(d_delta, 2 * hidden)
        nn.init.zeros_(self.mod.weight)
        nn.init.zeros_(self.mod.bias)

    def forward(self, x: torch.Tensor, demb: torch.Tensor) -> torch.Tensor:
        # F.rms_norm (fused, fp32 reduction inside, no explicit fp32 tensor
        # retained for backward) + fused addcmul: memory-lean FiLM.
        # (2026-09-14 OOM lesson: the naive chain h*(1+g)+b with a python-level
        # float() RMSNorm retained ~5GB extra autograd intermediates at
        # ~130k queries/step and OOMed the 32GB V100.)
        h = self.fc(F.rms_norm(x, (x.shape[-1],), self.norm.weight.to(x.dtype), self.norm.eps))
        g, b = self.mod(demb).chunk(2, dim=-1)
        return F.silu(torch.addcmul(b, h, g).add(h))  # h*(1+g)+b, fewer intermediates


class FiLMByteHead(nn.Module):
    """FiLM-conditioned byte head: delta_emb drives per-block modulation."""

    def __init__(
        self,
        d_in: int = 768,
        k_max: int = 48,
        d_delta: int = 64,
        hidden: int = 1536,
        depth: int = 4,
        film_layers: int = 2,
    ):
        super().__init__()
        self.delta_emb = nn.Embedding(k_max, d_delta)
        self.in_proj = nn.Linear(d_in, hidden)
        self.films = nn.ModuleList(
            [FiLMBlock(hidden, d_delta) for _ in range(film_layers)]
        )
        self.rest = nn.ModuleList(
            [nn.Linear(hidden, hidden) for _ in range(depth - 1 - film_layers)]
        )
        self.out = nn.Linear(hidden, 256)

    def forward(self, h: torch.Tensor, delta_idx: torch.Tensor) -> torch.Tensor:
        """h (N, d_in), delta_idx (N,) long in [0, k_max) -> (N, 256)."""
        demb = self.delta_emb(delta_idx)
        x = F.silu(self.in_proj(h))
        for blk in self.films:
            x = blk(x, demb)
        for fc in self.rest:
            x = F.silu(fc(x))
        return self.out(x)
