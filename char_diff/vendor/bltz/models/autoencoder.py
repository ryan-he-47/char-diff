"""v2 byte-string autoencoder (docs/31 §2.2): reconstruction-shaped embedding.

Eθ: byte emb (d_byte) + within-patch position CONCAT (d_pos — order must be
    visible; 2026-09-17 user reminder) -> ISAB x layers -> PMA(1) -> LN ->
    Linear(width -> d_emb). Trained ONLY by reconstruction.
Dθ: conditional query decoder (λ, Δ) -> 257-way softmax (256 bytes + EOS,
    EOS = row 256). FiLM conditioning (adaLN-Zero kin: zero-init conditioner).
    Δ ∈ 1..l_max+1; Δ = len+1 queries EOS. Trained ONLY by reconstruction.

Backbone gradients NEVER touch these (hard detach at the v2 model boundary).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder import ISAB, PMA
from .head import FiLMBlock

EOS_ID = 256  # 257th softmax row


class ByteStringEncoder(nn.Module):
    def __init__(
        self,
        d_byte: int = 192,
        d_pos: int = 64,
        width: int = 256,
        layers: int = 2,
        heads: int = 4,
        inducing: int = 8,
        l_max: int = 32,
        d_emb: int = 32,
    ):
        super().__init__()
        self.byte_emb = nn.Embedding(256, d_byte)
        self.pos_emb = nn.Embedding(l_max, d_pos)
        self.in_proj = nn.Linear(d_byte + d_pos, width)
        if inducing and inducing > 0:
            self.layers = nn.ModuleList(
                [ISAB(width, heads, inducing) for _ in range(layers)]
            )
        else:
            # standard self-attention (2026-09-29 user spec: no inducing
            # points; pre-norm transformer blocks, everything else unchanged)
            self.layers = nn.ModuleList([
                nn.TransformerEncoderLayer(
                    width, heads, dim_feedforward=4 * width,
                    dropout=0.0, activation="gelu", batch_first=True,
                    norm_first=True,
                )
                for _ in range(layers)
            ])
        self.pool = PMA(width, heads, k=1)
        self.out_norm = nn.LayerNorm(width)
        self.out_proj = nn.Linear(width, d_emb)

    def forward(self, byte_ids: torch.Tensor, pad_mask: torch.Tensor) -> torch.Tensor:
        """byte_ids (B, L) long in [0,256), pad_mask (B, L) True=pad -> (B, d_emb)."""
        B, L = byte_ids.shape
        pos = torch.arange(L, device=byte_ids.device).expand(B, L)
        x = torch.cat([self.byte_emb(byte_ids.clamp(0, 255)), self.pos_emb(pos)], dim=-1)
        x = self.in_proj(x)
        for lyr in self.layers:
            if isinstance(lyr, nn.TransformerEncoderLayer):
                x = lyr(x, src_key_padding_mask=pad_mask)
            else:
                x = lyr(x, key_padding_mask=pad_mask)
        h = self.pool(x, key_padding_mask=pad_mask)[:, 0]
        return self.out_proj(self.out_norm(h))


class ByteStringDecoder(nn.Module):
    """FiLM-conditioned (λ, Δ) -> 257 logits. Mirrors FiLMByteHead, EOS-extended."""

    def __init__(
        self,
        d_emb: int = 32,
        l_max: int = 32,
        d_delta: int = 64,
        hidden: int = 256,
        depth: int = 3,
        film_layers: int = 2,
    ):
        super().__init__()
        self.k_max = l_max + 1  # Δ = 1..l_max bytes, Δ = l_max+1 slot for EOS queries
        self.delta_emb = nn.Embedding(self.k_max, d_delta)
        self.in_proj = nn.Linear(d_emb, hidden)
        self.films = nn.ModuleList(
            [FiLMBlock(hidden, d_delta) for _ in range(film_layers)]
        )
        self.rest = nn.ModuleList(
            [nn.Linear(hidden, hidden) for _ in range(depth - 1 - film_layers)]
        )
        self.out = nn.Linear(hidden, 257)

    def forward(self, lam: torch.Tensor, delta_idx: torch.Tensor) -> torch.Tensor:
        """lam (N, d_emb), delta_idx (N,) long in [0, k_max) -> (N, 257)."""
        demb = self.delta_emb(delta_idx)
        x = F.silu(self.in_proj(lam))
        for blk in self.films:
            x = blk(x, demb)
        for fc in self.rest:
            x = F.silu(fc(x))
        return self.out(x)


class ByteStringAE(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        m = cfg.model
        self.l_max = int(m.l_max)
        self.encoder = ByteStringEncoder(
            d_byte=m.d_byte, d_pos=m.d_pos, width=m.enc_width,
            layers=m.enc_layers, heads=m.enc_heads, inducing=m.inducing,
            l_max=m.l_max, d_emb=m.d_emb,
        )
        self.decoder = ByteStringDecoder(
            d_emb=m.d_emb, l_max=m.l_max, d_delta=m.d_delta,
            hidden=m.dec_hidden, depth=m.dec_depth, film_layers=m.film_layers,
        )

    def forward(self, byte_ids: torch.Tensor, pad_mask: torch.Tensor) -> torch.Tensor:
        """(B, L) -> logits (B, l_max+1, 257); query Δ=d+1 at index d."""
        B = byte_ids.shape[0]
        lam = self.encoder(byte_ids, pad_mask)
        K = self.l_max + 1
        lam_q = lam.repeat_interleave(K, dim=0)
        d_idx = torch.arange(K, device=byte_ids.device).repeat(B)
        return self.decoder(lam_q, d_idx).view(B, K, 257)

    @torch.no_grad()
    def decode(self, lam: torch.Tensor) -> list[list[int]]:
        """λ (N, d_emb) -> byte lists (EOS-truncated). Greedy argmax per Δ."""
        N = lam.shape[0]
        K = self.l_max + 1
        d_idx = torch.arange(K, device=lam.device).repeat(N)
        logits = self.decoder(lam.repeat_interleave(K, dim=0), d_idx)
        pred = logits.argmax(-1).view(N, K).cpu().tolist()
        out: list[list[int]] = []
        for row in pred:
            bs: list[int] = []
            for v in row:
                if v == EOS_ID or len(bs) >= self.l_max:
                    break  # EOS stop; byte at Δ=l_max+1 is invalid -> truncate
                bs.append(v)
            out.append(bs)
        return out
