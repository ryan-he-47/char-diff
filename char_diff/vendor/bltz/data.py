"""Data pipeline: texts -> segmented patch sequences -> batched tensors.

Batch format (S = n_patches per sequence, L = l_max):
  byte_ids   (B, S, L)   long, PAD_ID = 256 for padded positions
  pad_mask   (B, S, L)   bool, True = padded
  flat       (B, S*L)    long, concatenated patch bytes; positions >= flat_len are 0
  flat_len   (B,)        long
  ends       (B, S)      long, EXCLUSIVE end offsets: patch j occupies
                         flat[ends[j-1]:ends[j]] (ends[-1] := 0)
  patch_pos  (B, S*L)    long, patch index (0..S-1) of each flat position

v1 simplification (documented in docs/02 Stage 1): the patch stream is built by
concatenating documents; sequences may cross document boundaries and the
backbone uses a plain causal mask. Document-boundary block-causal masking is a
v2 refinement (BLT parity).
"""
from __future__ import annotations

import random
from collections.abc import Iterator

import torch

from .segment import Segmenter

PAD_ID = 256


def tensorize_units(flat_bytes: torch.Tensor, lens: torch.Tensor, l_max: int) -> dict[str, torch.Tensor]:
    """One sequence from a unit stream -> batch dict entry (no batch dim).

    flat_bytes (F,) uint8-compatible, lens (S,) patch lengths. Vectorized:
    scatter each flat byte to (patch_idx, pos_in_patch) in a padded (S, l_max)
    canvas."""
    S = int(lens.numel())
    F = int(flat_bytes.numel())
    dev = flat_bytes.device
    ends = lens.cumsum(0)
    patch_pos = torch.repeat_interleave(torch.arange(S, device=dev), lens)
    pos_in = torch.arange(F, device=dev) - (ends - lens)[patch_pos]

    byte_ids = torch.full((S, l_max), PAD_ID, dtype=torch.long, device=dev)
    byte_ids[patch_pos, pos_in] = flat_bytes.long()
    pad_mask = torch.ones(S, l_max, dtype=torch.bool, device=dev)
    pad_mask[patch_pos, pos_in] = False

    flat = torch.zeros(S * l_max, dtype=torch.long, device=dev)
    flat[:F] = flat_bytes.long()
    pp = torch.zeros(S * l_max, dtype=torch.long, device=dev)
    pp[:F] = patch_pos
    return {
        "byte_ids": byte_ids,
        "pad_mask": pad_mask,
        "flat": flat,
        "flat_len": torch.tensor(F, dtype=torch.long),
        "ends": ends,
        "patch_pos": pp,
    }


def collate_sequences(seqs: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """Stack per-sequence dicts into a batch dict (uniform shapes by construction)."""
    return {k: torch.stack([s[k] for s in seqs]) for k in seqs[0]}


def build_batch(
    texts: list[str],
    segmenter: Segmenter,
    n_patches: int,
    l_max: int,
    augment: bool = False,
    rng: random.Random | None = None,
) -> dict[str, torch.Tensor]:
    """Segment texts and pack consecutive patches into S-length sequences."""
    stream: list[str] = []
    for t in texts:
        stream.extend(segmenter.segment_str(t, rng=rng, augment=augment))
    n_seq = len(stream) // n_patches
    if n_seq == 0:
        raise ValueError(f"not enough patches ({len(stream)}) for n_patches={n_patches}")

    B = n_seq
    byte_ids = torch.full((B, n_patches, l_max), PAD_ID, dtype=torch.long)
    pad_mask = torch.ones(B, n_patches, l_max, dtype=torch.bool)
    flat = torch.zeros(B, n_patches * l_max, dtype=torch.long)
    flat_len = torch.zeros(B, dtype=torch.long)
    ends = torch.zeros(B, n_patches, dtype=torch.long)
    patch_pos = torch.zeros(B, n_patches * l_max, dtype=torch.long)

    for b in range(B):
        pos = 0
        for j, unit in enumerate(stream[b * n_patches : (b + 1) * n_patches]):
            ub = unit.encode("utf-8")
            L = len(ub)
            assert L <= l_max, f"unit longer than l_max: {unit!r}"
            byte_ids[b, j, :L] = torch.tensor(list(ub), dtype=torch.long)
            pad_mask[b, j, :L] = False
            flat[b, pos : pos + L] = byte_ids[b, j, :L]
            patch_pos[b, pos : pos + L] = j
            pos += L
            ends[b, j] = pos
        flat_len[b] = pos

    return {
        "byte_ids": byte_ids,
        "pad_mask": pad_mask,
        "flat": flat,
        "flat_len": flat_len,
        "ends": ends,
        "patch_pos": patch_pos,
    }


def stream_fineweb_texts(
    n_docs: int, name: str = "sample-10BT", split: str = "train"
) -> Iterator[str]:
    """Stream the first n_docs texts from FineWeb-Edu (ungated, ODC-By)."""
    from datasets import load_dataset

    ds = load_dataset(
        "HuggingFaceFW/fineweb-edu", name=name, split=split, streaming=True
    )
    for i, row in enumerate(ds):
        if i >= n_docs:
            break
        yield row["text"]
