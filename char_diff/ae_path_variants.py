"""Exploratory costs and exact-K controls on the frozen UTF-8 candidate DAG.

No training/model changes. All paths retain complete byte+EOS reconstruction.
Boundary prior disagreement is a symmetric difference, including both inserted
cuts and preferred cuts crossed by an edge. Exact-K controls separate its effect
on boundaries from its effect on compression.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import random

import numpy as np

from .ae_paths import NoFeasiblePath, boundary_offsets, candidate_spans


def quality(row, kind):
    nll = float(row['target_nll_nats'])
    if kind == 'none':
        return 0.0
    if kind == 'nll':
        return nll
    if kind == 'entropy':
        return float(row['entropy_nats'])
    if kind == 'byte_risk':
        return row['bytes'] * -math.expm1(-nll)
    if kind == 'margin_risk':
        x = -float(row['min_logit_margin'])
        return row['bytes'] * (max(x, 0.0) + math.log1p(math.exp(-abs(x))))
    raise ValueError(f'Unknown quality kind {kind}')


def default_variants():
    out = [{'name': 'min_k', 'quality': 'none', 'weight': 0.0}]
    for kind in ['nll', 'entropy', 'byte_risk', 'margin_risk']:
        for weight in [.1, 1., 10.]:
            out.append({'name': f'{kind}_{weight:g}', 'quality': kind, 'weight': weight})
    for prior in ['regex', 'bpe']:
        for eta in [.1, .5, 1.]:
            out.append({'name': f'nll_1_{prior}_{eta:g}', 'quality': 'nll', 'weight': 1.,
                        'prior': prior, 'prior_weight': eta})
        out.append({'name': f'nll_1_{prior}_0.5_samek', 'quality': 'nll', 'weight': 1.,
                    'prior': prior, 'prior_weight': .5, 'same_k_as': 'nll_1'})
    out.append({'name': 'random_samek_nll_1', 'quality': 'none', 'weight': 0.,
                'same_k_as': 'nll_1', 'random': True})
    return out


@dataclass
class Edge:
    source: int
    target: int
    start: int
    end: int
    unit: bytes
    score: dict


class CandidateGraph:
    def __init__(self, raw, scores, *, max_bytes=32):
        self.raw = raw
        self.offsets = boundary_offsets(raw)
        node = {offset: i for i, offset in enumerate(self.offsets)}
        self.edges = []
        self.incoming = [[] for _ in self.offsets]
        self.outgoing = [[] for _ in self.offsets]
        self.candidate_occurrences = 0
        self.feasible_occurrences = 0
        for start, end in candidate_spans(raw, max_bytes=max_bytes):
            row = scores[raw[start:end]]
            self.candidate_occurrences += 1
            if not row['feasible']:
                continue
            self.feasible_occurrences += 1
            edge = Edge(node[start], node[end], start, end, raw[start:end], row)
            self.incoming[edge.target].append(len(self.edges))
            self.outgoing[edge.source].append(len(self.edges))
            self.edges.append(edge)
        # Sorted starts make vectorized exact-cost ties deterministic.
        for incoming in self.incoming:
            incoming.sort(key=lambda i: self.edges[i].start)
        width = max(map(len, self.incoming), default=0)
        self.matrix_edges = np.full((len(self.offsets), max(width, 1)), -1, dtype=np.int64)
        self.matrix_sources = np.zeros_like(self.matrix_edges)
        for j, incoming in enumerate(self.incoming):
            for k, edge_id in enumerate(incoming):
                self.matrix_edges[j, k] = edge_id
                self.matrix_sources[j, k] = self.edges[edge_id].source
        self._qualities = {}
        self._prior_costs = {}

    def edge_costs(self, spec, prior_boundaries=()):
        prior = sorted(set(prior_boundaries) - {0, len(self.raw)})
        legal = set(self.offsets)
        if any(b not in legal for b in prior):
            raise ValueError('Prior boundaries must be valid UTF-8 endpoints')
        weight, eta = float(spec.get('weight', 0)), float(spec.get('prior_weight', 0))
        if not all(math.isfinite(v) and v >= 0 for v in [weight, eta]):
            raise ValueError('Cost weights must be finite and nonnegative')
        kind = spec['quality']
        if kind not in self._qualities:
            self._qualities[kind] = np.asarray([quality(e.score, kind) for e in self.edges], dtype=np.float64)
        values = 1. + weight * self._qualities[kind]
        if spec.get('prior'):
            prior_key = tuple(prior)
            if prior_key not in self._prior_costs:
                starts = np.asarray([e.start for e in self.edges], dtype=np.int64)
                ends = np.asarray([e.end for e in self.edges], dtype=np.int64)
                missing = np.searchsorted(prior, ends, side='left') - np.searchsorted(prior, starts, side='right')
                extra = (ends != len(self.raw)) & ~np.isin(ends, prior)
                self._prior_costs[prior_key] = missing + extra
            values = values + eta * self._prior_costs[prior_key]
        if not np.isfinite(values).all():
            raise ValueError('Non-finite path costs')
        return values

    def solve(self, spec, *, prior_boundaries=(), exact_k=None):
        costs = self.edge_costs(spec, prior_boundaries)
        n = len(self.offsets)
        if exact_k is None:
            distance = [math.inf] * n
            counts = [n] * n
            previous = [-1] * n
            distance[0], counts[0] = 0., 0
            for target in range(1, n):
                best = None
                for edge_id in self.incoming[target]:
                    edge = self.edges[edge_id]
                    if math.isfinite(distance[edge.source]):
                        key = (distance[edge.source] + costs[edge_id], counts[edge.source] + 1, edge.start)
                        if best is None or key < best:
                            best = key
                            previous[target] = edge_id
                if best is not None:
                    distance[target], counts[target] = best[:2]
            if not math.isfinite(distance[-1]):
                raise NoFeasiblePath(f'No feasible path to {len(self.raw)} bytes')
            selected, node = [], n - 1
            while node:
                edge_id = previous[node]
                selected.append(edge_id)
                node = self.edges[edge_id].source
            return self.result(list(reversed(selected)), costs)
        if exact_k < 0 or exact_k > n - 1:
            raise NoFeasiblePath(f'Impossible requested K={exact_k}')
        matrix = np.full_like(self.matrix_sources, np.inf, dtype=np.float64)
        valid = self.matrix_edges >= 0
        matrix[valid] = costs[self.matrix_edges[valid]]
        previous = np.full((exact_k + 1, n), -1, dtype=np.int64)
        distance = np.full(n, np.inf)
        distance[0] = 0
        for k in range(1, exact_k + 1):
            candidate = distance[self.matrix_sources] + matrix
            choices = candidate.argmin(axis=1)
            distance = candidate[np.arange(n), choices]
            previous[k] = self.matrix_edges[np.arange(n), choices]
            previous[k, ~np.isfinite(distance)] = -1
        if not math.isfinite(distance[-1]):
            raise NoFeasiblePath(f'No feasible path with K={exact_k}')
        selected, node = [], n - 1
        for k in range(exact_k, 0, -1):
            edge_id = int(previous[k, node])
            selected.append(edge_id)
            node = self.edges[edge_id].source
        assert node == 0
        return self.result(list(reversed(selected)), costs)

    def random_fixed_k(self, exact_k, seed):
        """Uniform over feasible paths with exactly K edges, using integer counts."""
        n = len(self.offsets)
        ways = [[0] * n for _ in range(exact_k + 1)]
        ways[0][-1] = 1
        for k in range(1, exact_k + 1):
            prev, current = ways[k - 1], ways[k]
            for node in range(n - 2, -1, -1):
                current[node] = sum(prev[self.edges[i].target] for i in self.outgoing[node])
        if not ways[exact_k][0]:
            raise NoFeasiblePath(f'No feasible random path with K={exact_k}')
        rng, selected, node = random.Random(seed), [], 0
        for k in range(exact_k, 0, -1):
            pick = rng.randrange(ways[k][node])
            for edge_id in self.outgoing[node]:
                edge = self.edges[edge_id]
                count = ways[k - 1][edge.target]
                if pick < count:
                    selected.append(edge_id)
                    node = edge.target
                    break
                pick -= count
        assert node == n - 1
        result = self.result(selected, np.ones(len(self.edges)))
        result['uniform_exact_k_paths_log_count'] = math.log(ways[exact_k][0])
        return result

    def result(self, selected, costs):
        edges = [self.edges[i] for i in selected]
        units = []
        for edge in edges:
            units.append({'start': edge.start, 'end': edge.end, 'hex': edge.unit.hex(),
                          'bytes': len(edge.unit), **edge.score})
        assert b''.join(edge.unit for edge in edges) == self.raw
        return {'coverage': True, 'k': len(edges), 'cost': float(sum(costs[i] for i in selected)),
                'boundaries': [0] + [edge.end for edge in edges], 'units': units,
                'total_nll': sum(float(e.score['target_nll_nats']) for e in edges),
                'total_byte_nll': sum(float(e.score.get('target_byte_nll_nats', e.score['target_nll_nats'])) for e in edges),
                'total_eos_nll': sum(float(e.score.get('eos_nll_nats', 0.)) for e in edges),
                'total_entropy': sum(float(e.score['entropy_nats']) for e in edges),
                'byte_risk_sum': sum(quality(e.score, 'byte_risk') for e in edges),
                'margin_risk_sum': sum(quality(e.score, 'margin_risk') for e in edges)}
