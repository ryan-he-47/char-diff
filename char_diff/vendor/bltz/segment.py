"""Fixed semantic segmenter (docs/01-design.md §5.1, decision D3).

Deterministic base segmentation — word core + TRAILING delimiters:
  * latin word core: ``[0-9A-Za-z]+`` with optional trailing contraction
    (``'s`` ``'t`` ``'re`` ...), e.g. ``don't`` stays one core;
  * any other single word char (CJK ideographs, Cyrillic, Greek, underscore,
    digits outside latin runs, ...) is its own core;
  * delimiters = everything else (punctuation, symbols, whitespace); a maximal
    run of delimiters immediately FOLLOWING a core is attached to that core,
    so a base unit looks like ``"Hello, "`` or ``"world!"``; a leading
    delimiter run with no preceding core becomes its own unit;
  * units longer than ``l_max`` BYTES are subdivided greedily at character
    boundaries (first-fit; deliberately no BPE fallback table — design axiom:
    the architecture must not depend on a precise tokenizer).

Random augmentation (training only, resampled per epoch):
  * ``p_merge``: merge two adjacent units whose combined byte length <= l_max;
  * ``p_split``: split one unit at a random character boundary (needs >= 2 chars).

All cuts happen at str character boundaries, so every emitted chunk is valid
UTF-8 and ``b"".join(chunks) == original_bytes`` always holds (roundtrip).

The segmentation is intentionally crude (design axiom + user's stance): it only
needs to give the encoder semantically "complete-ish" units on average; the
random augmentation additionally teaches the encoder to stay robust on the
arbitrary byte spans the inference-time entropy stop will produce.
"""
from __future__ import annotations

import random

import regex as re

_LATIN = r"[0-9A-Za-z]+(?:'[0-9A-Za-z]{1,2})?"
_OTHER = r"(?![0-9A-Za-z])\w"
_DELIM = r"(?:[^\w\s]|\s)"
_UNIT = re.compile(
    rf"(?:{_LATIN}|{_OTHER})(?:{_DELIM}*)(?={_LATIN}|{_OTHER}|$)|(?:{_DELIM}+)"
)


def _split_long(unit: str, l_max: int) -> list[str]:
    """Greedy first-fit split of a unit into <= l_max-byte chunks (char-safe)."""
    if len(unit.encode("utf-8")) <= l_max:
        return [unit]
    chunks: list[str] = []
    cur: list[str] = []
    cur_bytes = 0
    for ch in unit:
        ch_bytes = len(ch.encode("utf-8"))
        if cur and cur_bytes + ch_bytes > l_max:
            chunks.append("".join(cur))
            cur, cur_bytes = [], 0
        cur.append(ch)
        cur_bytes += ch_bytes
    if cur:
        chunks.append("".join(cur))
    return chunks


class Segmenter:
    """Deterministic base segmentation + optional random augmentation."""

    def __init__(self, l_max: int = 16, p_split: float = 0.0, p_merge: float = 0.0):
        if l_max < 4:
            raise ValueError("l_max must be >= 4 (a UTF-8 char can be 4 bytes)")
        self.l_max = l_max
        self.p_split = p_split
        self.p_merge = p_merge

    def units(self, text: str) -> list[str]:
        """Raw regex units (before long-unit subdivision)."""
        return _UNIT.findall(text)

    def segment_str(
        self, text: str, rng: random.Random | None = None, augment: bool = False
    ) -> list[str]:
        """Deterministic base segmentation; augment=True applies p_merge/p_split."""
        out: list[str] = []
        for u in self.units(text):
            out.extend(_split_long(u, self.l_max))
        if augment:
            if rng is None:
                raise ValueError("augment=True requires an rng")
            out = self._augment(out, rng)
        return out

    def segment_bytes(
        self, data: bytes, rng: random.Random | None = None, augment: bool = False
    ) -> list[bytes]:
        text = data.decode("utf-8")
        return [u.encode("utf-8") for u in self.segment_str(text, rng, augment)]

    def _augment(self, units: list[str], rng: random.Random) -> list[str]:
        # merge pass: randomly merge adjacent units (combined <= l_max bytes)
        merged: list[str] = []
        i = 0
        while i < len(units):
            if (
                i + 1 < len(units)
                and rng.random() < self.p_merge
                and len(units[i].encode("utf-8")) + len(units[i + 1].encode("utf-8"))
                <= self.l_max
            ):
                merged.append(units[i] + units[i + 1])
                i += 2
            else:
                merged.append(units[i])
                i += 1
        # split pass: randomly cut a unit at a random char boundary
        out: list[str] = []
        for u in merged:
            if rng.random() < self.p_split and len(u) >= 2:
                cut = rng.randrange(1, len(u))
                out.append(u[:cut])
                out.append(u[cut:])
            else:
                out.append(u)
        return out


def augment_unit_bytes(
    units: list[bytes],
    l_max: int,
    p_split: float,
    p_merge: float,
    rng: random.Random,
) -> list[bytes]:
    """Same augmentation as Segmenter._augment but on cached byte units
    (read-time path of the disk cache). Byte-concat merges are always
    UTF-8 safe; splits decode the (<= l_max-byte) unit to find a char boundary."""
    merged: list[bytes] = []
    i = 0
    while i < len(units):
        if (
            i + 1 < len(units)
            and rng.random() < p_merge
            and len(units[i]) + len(units[i + 1]) <= l_max
        ):
            merged.append(units[i] + units[i + 1])
            i += 2
        else:
            merged.append(units[i])
            i += 1
    out: list[bytes] = []
    for ub in merged:
        s = ub.decode("utf-8")
        if rng.random() < p_split and len(s) >= 2:
            cut = rng.randrange(1, len(s))
            out.append(s[:cut].encode("utf-8"))
            out.append(s[cut:].encode("utf-8"))
        else:
            out.append(ub)
    return out
