"""Quality-weighted paths over reconstructible local byte strings.

The production boundary mode is ``utf8``: every endpoint is a Unicode code
point boundary, while ``max_bytes`` still limits encoded byte length. ``byte``
is available for algorithm comparisons. Scores belong to one fixed AE and
scoring configuration; callers must keep caches separate across configurations.

This module does no model loading. The scorer must check exact reconstruction,
including the target EOS position, when setting CandidateScore.feasible.
"""
from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, MutableMapping, Sequence
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class CandidateScore:
    feasible: bool
    quality: float = 0.0

    def __post_init__(self):
        quality = float(self.quality)
        if self.feasible and (not math.isfinite(quality) or quality < 0):
            raise ValueError("Feasible candidates need finite, nonnegative quality")
        object.__setattr__(self, "quality", quality)


@dataclass(frozen=True)
class PathResult:
    boundaries: tuple[int, ...]
    units: tuple[bytes, ...]
    cost: float
    quality_sum: float


class NoFeasiblePath(ValueError):
    """The supplied candidate graph has no path covering the full input."""


BatchScorer = Callable[[Sequence[bytes]], Sequence[CandidateScore]]


def boundary_offsets(raw: bytes, mode: str = "utf8") -> tuple[int, ...]:
    """Return byte offsets, including 0 and len(raw), without changing bytes.

    Strict UTF-8 decoding rejects incomplete sequences, overlong encodings,
    surrogate code points and values beyond U+10FFFF. Combining code points
    remain separate endpoints; this is a code-point, not a grapheme rule.
    """
    if not isinstance(raw, bytes):
        raise TypeError("raw must be bytes")
    if mode == "byte":
        return tuple(range(len(raw) + 1))
    if mode != "utf8":
        raise ValueError("boundary mode must be 'utf8' or 'byte'")
    raw.decode("utf-8", errors="strict")
    return tuple(i for i, value in enumerate(raw) if value & 0xC0 != 0x80) + (len(raw),)


def _check_max_bytes(max_bytes: int) -> None:
    if isinstance(max_bytes, bool) or not isinstance(max_bytes, int) or max_bytes < 1:
        raise ValueError("max_bytes must be a positive integer")


def _spans(offsets: tuple[int, ...], max_bytes: int) -> Iterator[tuple[int, int]]:
    for i in range(len(offsets) - 1):
        start = offsets[i]
        for end_index in range(i + 1, len(offsets)):
            end = offsets[end_index]
            if end - start > max_bytes:
                break
            yield start, end


def candidate_spans(raw: bytes, *, max_bytes: int = 32,
                    boundary_mode: str = "utf8") -> Iterator[tuple[int, int]]:
    """Enumerate every allowed edge in the full input, in start/end order."""
    _check_max_bytes(max_bytes)
    yield from _spans(boundary_offsets(raw, boundary_mode), max_bytes)


def score_candidates(raw: bytes, scorer: BatchScorer, *, max_bytes: int = 32,
                     boundary_mode: str = "utf8", batch_size: int = 512,
                     cache: MutableMapping[bytes, CandidateScore] | None = None
                     ) -> dict[bytes, CandidateScore]:
    """Score unique candidate byte strings once, using bounded scorer batches.

    The returned dictionary holds scores for this input, not latent vectors.
    ``cache`` may be an externally bounded LRU; it must be scoped to the same
    model/configuration. Deduplication also covers repeated uncached substrings.
    """
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    scores: dict[bytes, CandidateScore] = {}
    pending: list[bytes] = []
    pending_set: set[bytes] = set()

    def flush() -> None:
        values = list(scorer(tuple(pending)))
        if len(values) != len(pending):
            raise ValueError("Scorer must return one CandidateScore per input")
        if any(not isinstance(value, CandidateScore) for value in values):
            raise TypeError("Scorer outputs must be CandidateScore objects")
        for unit, score in zip(pending, values):
            scores[unit] = score
            if cache is not None:
                cache[unit] = score
        pending.clear()
        pending_set.clear()

    for start, end in candidate_spans(raw, max_bytes=max_bytes, boundary_mode=boundary_mode):
        unit = raw[start:end]
        if unit in scores or unit in pending_set:
            continue
        if cache is not None and unit in cache:
            value = cache[unit]
            if not isinstance(value, CandidateScore):
                raise TypeError("Cache values must be CandidateScore objects")
            scores[unit] = value
        else:
            pending.append(unit)
            pending_set.add(unit)
            if len(pending) == batch_size:
                flush()
    if pending:
        flush()
    return scores


def shortest_path(raw: bytes, scores: Mapping[bytes, CandidateScore] | None = None, *,
                  scorer: BatchScorer | None = None, quality_weight: float = 1.0,
                  max_bytes: int = 32, boundary_mode: str = "utf8",
                  batch_size: int = 512,
                  cache: MutableMapping[bytes, CandidateScore] | None = None) -> PathResult:
    """Minimize K + quality_weight * sum(Q) over all feasible paths.

    Supply exactly one of ``scores`` or ``scorer``. A missing map entry is an
    error, not an implicitly rejected edge. Python floats provide binary64
    accumulation. Exact numeric ties choose fewer units, then the smaller
    predecessor byte offset, with the already-fixed prefix path retained.
    """
    _check_max_bytes(max_bytes)
    weight = float(quality_weight)
    if not math.isfinite(weight) or weight < 0:
        raise ValueError("quality_weight must be finite and nonnegative")
    if (scores is None) == (scorer is None):
        raise ValueError("Supply exactly one of scores or scorer")
    offsets = boundary_offsets(raw, boundary_mode)
    if scorer is not None:
        scores = score_candidates(raw, scorer, max_bytes=max_bytes,
                                  boundary_mode=boundary_mode, batch_size=batch_size, cache=cache)
    assert scores is not None
    count = len(offsets)
    costs = [math.inf] * count
    units_count = [count] * count
    previous = [-1] * count
    costs[0], units_count[0] = 0.0, 0
    for j in range(1, count):
        best = None
        for i in range(j - 1, -1, -1):
            if offsets[j] - offsets[i] > max_bytes:
                break
            unit = raw[offsets[i]:offsets[j]]
            score = scores[unit]
            if not isinstance(score, CandidateScore):
                raise TypeError("Score map values must be CandidateScore objects")
            if not score.feasible or (previous[i] < 0 and i != 0):
                continue
            cost = costs[i] + 1.0 + weight * score.quality
            if not math.isfinite(cost):
                raise ValueError("Path cost overflow; reduce quality score/weight scale")
            key = (cost, units_count[i] + 1, offsets[i])
            if best is None or key < best:
                best = key
                previous[j] = i
        if best is not None:
            costs[j], units_count[j] = best[:2]
    if count > 1 and previous[-1] < 0:
        reachable = max(offsets[i] for i in range(count) if math.isfinite(costs[i]))
        raise NoFeasiblePath(f"No feasible path to byte {len(raw)}; furthest reachable byte {reachable}")
    nodes = [count - 1]
    while nodes[-1] != 0:
        nodes.append(previous[nodes[-1]])
    boundaries = tuple(offsets[i] for i in reversed(nodes))
    units = tuple(raw[a:b] for a, b in zip(boundaries, boundaries[1:]))
    return PathResult(boundaries, units, costs[-1], math.fsum(scores[u].quality for u in units))
