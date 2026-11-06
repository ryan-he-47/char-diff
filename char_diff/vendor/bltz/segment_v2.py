"""v2 segmentation: enhanced BPE (docs/31 §2.1, 2026-09-17 拍板).

Two boundary sets per sample:
  1. rule pre-tokenizer: byte-class runs — space-like / word (ASCII alnum +
     bytes >= 0x80, i.e. utf8 continuations stay inside words) / symbol
     (ASCII punctuation);
  2. Qwen3.5 BPE tokenizer boundaries (tokenizer.json, local data dir).

Combine at every candidate gap (union of both sets): both agree -> always
split; only one -> split with probability p (0.5, baked once per sample —
single-pass training makes online re-rolling equivalent, 2026-09-17 拍板).
Units hard-capped at l_max bytes (long symbol runs are chopped).
"""
from __future__ import annotations

import os
import random
import unicodedata
from functools import lru_cache

import numpy as np

_SPACE = frozenset((9, 10, 11, 12, 13, 32))


def _byte_class(b: int) -> int:
    if b in _SPACE:
        return 0
    if b >= 0x80 or (0x30 <= b <= 0x39) or (0x41 <= b <= 0x5A) or (0x61 <= b <= 0x7A):
        return 1  # word (alnum + utf8 continuation)
    return 2  # symbol


_BYTE_CLASS = np.array([_byte_class(b) for b in range(256)], dtype=np.uint8)


def pre_boundaries(raw: bytes) -> set[int]:
    """Run edges of the byte-class pre-tokenizer (0 and len excluded).
    Vectorized: class lookup + diff (the pure-python per-byte loop was the
    build's bottleneck — 2026-09-20 local-build lesson)."""
    a = _BYTE_CLASS[np.frombuffer(raw, dtype=np.uint8)]
    if a.shape[0] < 2:
        return set()
    return set((np.nonzero(a[1:] != a[:-1])[0] + 1).tolist())

@lru_cache(maxsize=1)
def _tokenizer(tok_dir: str):
    from tokenizers import Tokenizer

    return Tokenizer.from_file(os.path.join(tok_dir, "tokenizer.json"))


def bpe_boundaries(raw: bytes, tok_dir: str) -> set[int]:
    """BPE token start offsets, mapped to BYTE offsets. Dirty-UTF8 fallback: {}."""
    try:
        s = raw.decode("utf-8", errors="surrogateescape")
        enc = _tokenizer(tok_dir).encode(s, add_special_tokens=False)
    except Exception:
        return set()
    # char idx -> byte idx; surrogateescape round-trips exactly. Vectorized:
    # ascii -> 1B; DC80-DCFF surrogates -> 1B (original byte); else utf8 width.
    ords = np.fromiter(map(ord, s), dtype=np.int64, count=len(s))
    widths = np.where(ords < 0x80, 1,
                      np.where((ords >= 0xDC80) & (ords <= 0xDCFF), 1,
                               np.where(ords < 0x800, 2,
                                        np.where(ords < 0x10000, 3, 4))))
    cum = np.concatenate([[0], np.cumsum(widths)])
    out: set[int] = set()
    for start, _end in enc.offsets:
        b = int(cum[start])
        if 0 < b < len(raw):
            out.add(b)
    return out


def _is_nonsymbol(ch: str) -> bool:
    """Unicode-level classification (2026-09-28 word-arm spec): letters and
    numbers are non-symbol (word runs); everything else — punctuation,
    whitespace, symbols, marks, surrogates — is a symbol run."""
    return unicodedata.category(ch)[0] in ("L", "N")


def segment_word(raw: bytes, tok_dir: str, l_max: int = 32,
                 merge: str = "prev") -> list[bytes]:
    """Whole-word-dominant segmentation (user spec v2, 2026-09-28 third arm):

    Unicode classes: digit / word(letter) / symbol.
      * every DIGIT char is its own patch (digit strings compress worst and
        whole-string prediction is hard — user rule 2);
      * letters form maximal word runs (whole words, CJK runs too);
      * symbols form maximal runs (pure separators / punct sequences);
      * a patch that is exactly ONE space merges into an adjacent word patch
        (merge="prev": trailing style 'Stock ' — user rule 1; merge="next":
        leading style ' Stock' — the AE's native enhanced-BPE style, see
        2026-09-28 bucket finding), never into a digit patch;
      * any patch over l_max bytes is split with Qwen BPE (tokens capped).
    Deterministic — no rng.
    """
    s = raw.decode("utf-8", errors="surrogateescape")

    def _cls(ch: str) -> str:
        c = unicodedata.category(ch)[0]
        return "D" if c == "N" else ("W" if c == "L" else "S")

    patches: list[str] = []
    cur: list[str] = []
    cur_cls: str | None = None
    for ch in s:
        cls = _cls(ch)
        if cls == "D":
            if cur:
                patches.append("".join(cur))
                cur = []
                cur_cls = None
            patches.append(ch)  # digit: always alone
        elif cls == cur_cls:
            cur.append(ch)
        else:
            if cur:
                patches.append("".join(cur))
            cur = [ch]
            cur_cls = cls
    if cur:
        patches.append("".join(cur))

    def _is_digit_patch(p: str) -> bool:
        return len(p) == 1 and _cls(p[0]) == "D"

    if merge == "next":
        out: list[str] = []
        i = 0
        while i < len(patches):
            p = patches[i]
            if p == " " and i + 1 < len(patches) and not _is_digit_patch(patches[i + 1]):
                out.append(" " + patches[i + 1])
                i += 2
            else:
                out.append(p)
                i += 1
        merged = out
    else:  # merge == "prev" (user rule 1 literal)
        merged = []
        for p in patches:
            if p == " " and merged and not _is_digit_patch(merged[-1]):
                merged[-1] = merged[-1] + " "
                continue
            merged.append(p)

    units: list[bytes] = []
    for run in merged:
        b = run.encode("utf-8", errors="surrogateescape")
        if not b:
            continue
        if len(b) <= l_max:
            units.append(b)
        else:
            cuts = sorted(bpe_boundaries(b, tok_dir)) or []
            pos = 0
            for g in cuts + [len(b)]:
                piece = b[pos:g]
                pos = g
                for i in range(0, len(piece), l_max):
                    if piece[i : i + l_max]:
                        units.append(piece[i : i + l_max])
    return units


def segment_v2(
    raw: bytes,
    rng: random.Random,
    tok_dir: str,
    p: float = 0.5,
    l_max: int = 32,
    pure_bpe: bool = False,
) -> list[bytes]:
    """Segment one sample into byte-string units.

    pure_bpe (2026-09-28 用户洞察): use ONLY Qwen BPE boundaries — fully
    deterministic (same word always segments the same way). The enhanced
    mode's p_disagree randomness made "whole word vs its subwords" an
    unpredictable branch at every position, wasting GMM capacity on
    duplicate representations of the same possibility.
    """
    bpe = bpe_boundaries(raw, tok_dir)
    if pure_bpe:
        cuts = sorted(bpe)
    else:
        pre = pre_boundaries(raw)
        cuts = []
        for g in sorted(pre | bpe):
            if g in pre and g in bpe:
                cuts.append(g)
            elif rng.random() < p:
                cuts.append(g)
    units: list[bytes] = []
    pos = 0
    for g in cuts + [len(raw)]:
        seg = raw[pos:g]
        pos = g
        for i in range(0, len(seg), l_max):  # hard cap
            if seg[i : i + l_max]:
                units.append(seg[i : i + l_max])
    return units


def segment_v2_batch(
    raws: list[bytes],
    rngs: list[random.Random],
    tok_dir: str,
    p: float = 0.5,
    l_max: int = 32,
    pure_bpe: bool = False,
) -> list[list[bytes]]:
    """Batch variant: one Rust-side encode_batch (releases GIL, uses threads)
    instead of per-doc encode calls. Falls back to the per-doc path on any
    decode/encode exception (dirty UTF-8 docs)."""
    try:
        strs = [r.decode("utf-8", errors="surrogateescape") for r in raws]
        encs = _tokenizer(tok_dir).encode_batch(strs, add_special_tokens=False)
        if len(encs) != len(raws):
            raise ValueError("encode_batch length mismatch")
    except Exception:
        return [segment_v2(r, rng, tok_dir, p, l_max, pure_bpe) for r, rng in zip(raws, rngs)]
    out: list[list[bytes]] = []
    for raw, enc, rng in zip(raws, encs, rngs):
        s = raw.decode("utf-8", errors="surrogateescape")
        ords = np.fromiter(map(ord, s), dtype=np.int64, count=len(s))
        widths = np.where(ords < 0x80, 1,
                          np.where((ords >= 0xDC80) & (ords <= 0xDCFF), 1,
                                   np.where(ords < 0x800, 2,
                                            np.where(ords < 0x10000, 3, 4))))
        cum = np.concatenate([[0], np.cumsum(widths)])
        bpe: set[int] = set()
        for start, _end in enc.offsets:
            b = int(cum[start])
            if 0 < b < len(raw):
                bpe.add(b)
        if pure_bpe:
            cuts = sorted(bpe)
        else:
            pre = pre_boundaries(raw)
            cuts = []
            for g in sorted(pre | bpe):
                if g in pre and g in bpe:
                    cuts.append(g)
                elif rng.random() < p:
                    cuts.append(g)
        units: list[bytes] = []
        pos = 0
        for g in cuts + [len(raw)]:
            seg = raw[pos:g]
            pos = g
            for i in range(0, len(seg), l_max):
                if seg[i : i + l_max]:
                    units.append(seg[i : i + l_max])
        out.append(units)
    return out
