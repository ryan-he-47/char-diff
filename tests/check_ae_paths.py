"""Targeted checks for UTF-8 candidate graphs and quality-weighted paths."""
from pathlib import Path
import itertools
import random
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from char_diff.ae_paths import (CandidateScore, NoFeasiblePath, boundary_offsets,
                                candidate_spans, score_candidates, shortest_path)


def exhaustive(raw, scores, max_bytes, weight):
    """Enumerate every boundary subset independently of the DAG solver."""
    offsets = boundary_offsets(raw)
    paths = []
    for take in itertools.product([False, True], repeat=max(0, len(offsets) - 2)):
        edges = (0,) + tuple(x for x, use in zip(offsets[1:-1], take) if use) + (len(raw),)
        units = tuple(raw[a:b] for a, b in zip(edges, edges[1:]))
        if any(len(u) > max_bytes or not scores[u].feasible for u in units):
            continue
        cost = sum(1.0 + weight * scores[u].quality for u in units)
        paths.append(((cost, len(units), tuple(reversed(edges[:-1]))), edges))
    return min(paths)[1] if paths else None


class PathChecks(unittest.TestCase):
    def test_exhaustive_short_graphs(self):
        rng = random.Random(1234)
        for raw in [b"ababac", "a汉🙂a".encode(), b"abcdefg"]:
            for cap in [3, 7]:
                scores = {raw[a:b]: CandidateScore(rng.random() > .2, rng.randrange(8) / 4)
                          for a, b in candidate_spans(raw, max_bytes=cap)}
                for weight in [0., .5, 2.]:
                    expected = exhaustive(raw, scores, cap, weight)
                    if expected is None:
                        with self.assertRaises(NoFeasiblePath):
                            shortest_path(raw, scores, max_bytes=cap, quality_weight=weight)
                    else:
                        result = shortest_path(raw, scores, max_bytes=cap, quality_weight=weight)
                        self.assertEqual(result.boundaries, expected)
                        self.assertEqual(b"".join(result.units), raw)
        self.assertEqual(shortest_path(b"", {}).boundaries, (0,))

    def test_quality_changes_count_and_crosses_byte32(self):
        raw = b"abcd"
        scores = {raw[a:b]: CandidateScore(True, 20) for a, b in candidate_spans(raw)}
        scores.update({b"abcd": CandidateScore(True, 10),
                       b"ab": CandidateScore(True, 0), b"cd": CandidateScore(True, 0)})
        self.assertEqual(shortest_path(raw, scores, quality_weight=0).units, (raw,))
        self.assertEqual(shortest_path(raw, scores).units, (b"ab", b"cd"))
        long_raw = b"a" * 40
        scores = {b"a" * n: CandidateScore(True, 0 if n == 20 else 10) for n in range(1, 33)}
        self.assertEqual(shortest_path(long_raw, scores).boundaries, (0, 20, 40))
        tied = {b"a" * n: CandidateScore(True, 0) for n in range(1, 4)}
        self.assertEqual(shortest_path(b"aaaa", tied, max_bytes=3).boundaries, (0, 1, 4))

    def test_utf8_boundaries_and_unreachable(self):
        raw = "A汉🙂e\u0301\0".encode()
        expected = (0, 1, 4, 8, 9, 11, 12)
        self.assertEqual(boundary_offsets(raw), expected)
        for a, b in candidate_spans(raw, max_bytes=4):
            self.assertIn(a, expected); self.assertIn(b, expected)
            raw[a:b].decode("utf-8", "strict")
        scores = {raw[a:b]: CandidateScore(True) for a, b in candidate_spans(raw, max_bytes=3)}
        with self.assertRaises(NoFeasiblePath):
            shortest_path(raw, scores, max_bytes=3)
        for bad in [b"\x80", b"\xc0\xaf", b"\xed\xa0\x80", b"\xf4\x90\x80\x80", b"\xf0\x9f"]:
            with self.assertRaises(UnicodeDecodeError): boundary_offsets(bad)
        self.assertEqual(boundary_offsets(b"\x80", "byte"), (0, 1))
        with self.assertRaises(KeyError): shortest_path(b"a", {})
        with self.assertRaises(ValueError): CandidateScore(True, float("nan"))

    def test_batch_scoring_dedup_and_cache(self):
        raw = "ab汉ab汉".encode()
        cache = {b"ab": CandidateScore(True, 0)}
        calls = []
        def scorer(batch):
            self.assertLessEqual(len(batch), 3)
            calls.extend(batch)
            return [CandidateScore(True, len(unit) / 4) for unit in batch]
        scores = score_candidates(raw, scorer, max_bytes=5, batch_size=3, cache=cache)
        expected = {raw[a:b] for a, b in candidate_spans(raw, max_bytes=5)}
        self.assertEqual(set(scores), expected)
        self.assertEqual(len(calls), len(set(calls)))
        self.assertNotIn(b"ab", calls)
        first = shortest_path(raw, scores, max_bytes=5)
        count = len(calls)
        second = shortest_path(raw, scorer=scorer, cache=cache, max_bytes=5, batch_size=3)
        self.assertEqual(first, second)
        self.assertEqual(len(calls), count)
        with self.assertRaises(ValueError): score_candidates(b"ab", lambda batch: [], batch_size=2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
