"""Causal transformer backbone over patch latents (docs/01-design.md §5.3).

Llama-style: RMSNorm, RoPE, SwiGLU FFN, pre-norm residuals, SDPA with
is_causal=True (flash path on sm89). No token embedding table anywhere —
input is the encoder's patch latents, already at d_model.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return (x.float() * scale).type_as(x) * self.weight


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Half-split (GPT-NeoX style) RoPE. x (B, nh, T, hd); cos/sin (T, hd/2).

    NOTE: the interleaved-pairs formulation (stack([o1, o2], -1)) was measured
    at 104 ms/call on this machine — the stride-2 interleaved cat is severely
    uncoalesced (67.7% of the whole backbone's CUDA time, see
    scripts/prof_backbone.py output in docs/06). Contiguous-half cat is the
    same rotation, coalesced. Computed in x.dtype to avoid fp32 temporaries."""
    hd2 = x.shape[-1] // 2
    x1, x2 = x[..., :hd2], x[..., hd2:]
    cos = cos[None, None].to(x.dtype)
    sin = sin[None, None].to(x.dtype)
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


def apply_rope_flat(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Same rotation for the flat xformers layout x (1, L, nh, hd); cos/sin
    (L, hd/2) — L = B*T, positions repeat per window (cos/repeat pre-tiled)."""
    hd2 = x.shape[-1] // 2
    x1, x2 = x[..., :hd2], x[..., hd2:]
    cos = cos[None, :, None].to(x.dtype)
    sin = sin[None, :, None].to(x.dtype)
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


class Block(nn.Module):
    def __init__(self, d_model: int, nhead: int, ffn_mult: int):
        super().__init__()
        self.nhead = nhead
        self.hd = d_model // nhead
        self.ln1 = RMSNorm(d_model)
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.o = nn.Linear(d_model, d_model, bias=False)
        self.ln2 = RMSNorm(d_model)
        d_ff = ffn_mult * d_model
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w3 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
                attn_bias=None) -> torch.Tensor:
        B, T, D = x.shape
        h = self.ln1(x)
        if attn_bias is not None and not torch.is_tensor(attn_bias):
            # xformers structured block-diagonal causal mask (2026-10-03 效率
            # 优化): flatten (B, T) into one sequence; no (B,1,S,S) matrix —
            # the kernel skips cross-block pairs (V100 bench 64ms vs 267ms
            # dense vs 127ms plain causal at production shapes, docs/39).
            from xformers.ops import memory_efficient_attention

            qkv = self.qkv(h).view(B * T, 3, self.nhead, self.hd)
            cosf = cos[:T].repeat(B, 1)
            sinf = sin[:T].repeat(B, 1)
            q = apply_rope_flat(qkv[:, 0].unsqueeze(0), cosf, sinf)
            k = apply_rope_flat(qkv[:, 1].unsqueeze(0), cosf, sinf)
            v = qkv[:, 2].unsqueeze(0).contiguous()  # tail slice is strided
            a = memory_efficient_attention(q, k, v, attn_bias=attn_bias)
            x = x + self.o(a.view(B, T, D))
        else:
            qkv = self.qkv(h).reshape(B, T, 3, self.nhead, self.hd).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]
            q = apply_rope(q, cos[:T], sin[:T])
            k = apply_rope(k, cos[:T], sin[:T])
            if attn_bias is None:
                a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            else:
                # dense (B,1,S,S) bias carries causality itself (2026-10-03
                # 文档分块掩码, portable fallback) — no is_causal here.
                a = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_bias)
            x = x + self.o(a.transpose(1, 2).reshape(B, T, D))
        h2 = self.ln2(x)
        return x + self.w2(F.silu(self.w1(h2)) * self.w3(h2))


class Backbone(nn.Module):
    """Patch-latent causal transformer. Input (B, T, d_model) -> (B, T, d_model)."""

    def __init__(
        self,
        d_model: int = 768,
        nhead: int = 12,
        layers: int = 12,
        ffn_mult: int = 4,
        max_len: int = 520,
        rope_base: float = 500000.0,
        grad_ckpt: bool = False,
    ):
        super().__init__()
        self.rope_base = rope_base
        self.hd = d_model // nhead
        self.grad_ckpt = grad_ckpt
        self.blocks = nn.ModuleList(
            [Block(d_model, nhead, ffn_mult) for _ in range(layers)]
        )
        self.final_norm = RMSNorm(d_model)
        cos, sin = self._build_rope(max_len)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

    def _build_rope(self, n: int):
        inv_freq = 1.0 / (
            self.rope_base ** (torch.arange(0, self.hd, 2).float() / self.hd)
        )
        t = torch.arange(n).float()
        freqs = torch.outer(t, inv_freq)
        return freqs.cos(), freqs.sin()

    def forward(self, x: torch.Tensor, attn_bias: torch.Tensor | None = None) -> torch.Tensor:
        T = x.shape[1]
        if T > self.rope_cos.shape[0]:  # lazy extension (inference growth)
            cos, sin = self._build_rope(T)
            self.rope_cos = cos.to(x.device)
            self.rope_sin = sin.to(x.device)
        cos = self.rope_cos.to(x.device)
        sin = self.rope_sin.to(x.device)
        if torch.is_tensor(attn_bias) and attn_bias.dtype != x.dtype:
            attn_bias = attn_bias.to(x.dtype)  # SDPA needs mask dtype == query
        for blk in self.blocks:
            if self.grad_ckpt and self.training:
                x = torch.utils.checkpoint.checkpoint(
                    blk, x, cos, sin, attn_bias, use_reentrant=False
                )
            else:
                x = blk(x, cos, sin, attn_bias)
        return self.final_norm(x)
