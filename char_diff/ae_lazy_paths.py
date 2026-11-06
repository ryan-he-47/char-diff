"""Exact lazy shortest paths using the nonnegative NLL edge-cost lower bound."""
from __future__ import annotations

import math
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

from .ae_paths import boundary_offsets, NoFeasiblePath, PathResult


class LazyNLLGraph:
    """Unknown edges cost 1; scored edges cost 1+lambda*NLL or infinity.

    Repeatedly score unknown strings on the optimistic shortest path. A path
    whose edges are all scored attains the graph's lower bound and is optimal.
    Multiple independent graphs may pool their pending strings into GPU batches.
    Scores must come from the same frozen model and numerical configuration.
    """

    def __init__(self, raw: bytes, *, weight=10.0, max_bytes=32, offsets=None):
        if not math.isfinite(weight) or weight < 0:
            raise ValueError("Expected finite nonnegative weight")
        if max_bytes < 1:
            raise ValueError("Expected positive max_bytes")
        self.raw, self.weight = raw, weight
        full_offsets = boundary_offsets(raw)
        self.offsets = full_offsets if offsets is None else tuple(offsets)
        if (not self.offsets or self.offsets[0] != 0 or self.offsets[-1] != len(raw)
                or tuple(sorted(set(self.offsets))) != self.offsets
                or not set(self.offsets).issubset(full_offsets)):
            raise ValueError("Offsets must be ordered UTF-8 boundaries including both endpoints")
        n = len(self.offsets)
        self.costs = np.full((n, max_bytes), np.inf, dtype=np.float64)
        self.units = [[] for _ in range(n)]
        self.occurrences = {}
        self.known = {}
        for j in range(1, n):
            for k in range(1, min(j, max_bytes) + 1):
                i = j-k
                if self.offsets[j]-self.offsets[i] > max_bytes:
                    break
                unit = raw[self.offsets[i]:self.offsets[j]]
                self.units[j].append(unit)
                self.occurrences.setdefault(unit, []).append((j,k-1))
                self.costs[j,k-1] = 1.0
        ends, slots = np.nonzero(np.isfinite(self.costs))
        sources = ends-slots-1
        order = np.lexsort((ends,sources))
        self.ends, self.slots, self.sources = ends[order],slots[order],sources[order]
        self.csr = csr_matrix((self.costs[self.ends,self.slots],
                               (self.sources,self.ends)), shape=(n,n))

    def update(self, scores):
        for unit, score in scores.items():
            positions = self.occurrences.get(unit)
            if positions is None:
                continue
            self.known[unit] = score
            cost = 1.0 + self.weight*score.quality if score.feasible else math.inf
            if score.feasible and not math.isfinite(cost):
                raise ValueError("Path cost overflow")
            for j,k in positions:
                self.costs[j,k] = cost

    def optimistic_path(self):
        n = len(self.offsets)
        if n == 1:
            return PathResult((0,),(),0.,0.), []
        weights = self.costs[self.ends,self.slots]
        self.csr.data[:] = weights
        distance = dijkstra(self.csr, directed=True, indices=0)
        if not np.isfinite(distance[-1]):
            raise NoFeasiblePath(f"No feasible path to byte {len(self.raw)}")
        # A second compiled shortest-path pass breaks exact distance ties by K.
        # Retain only edges attaining the first pass's distance at their end.
        tight = (distance[self.sources]+weights == distance[self.ends]) & np.isfinite(distance[self.ends])
        self.csr.data[:] = np.where(tight,1.,np.inf)
        count = dijkstra(self.csr, directed=True, indices=0)
        nodes = [n-1]
        while nodes[-1]:
            j=nodes[-1];k=len(self.units[j]);predecessors=np.arange(j-1,j-k-1,-1)
            valid=(distance[predecessors]+self.costs[j,:k]==distance[j]) & (count[predecessors]+1==count[j])
            choices=predecessors[valid]
            if not len(choices):raise RuntimeError('Inconsistent shortest-path reconstruction')
            nodes.append(int(choices[-1]))
        boundaries = tuple(self.offsets[i] for i in reversed(nodes))
        units = tuple(self.raw[a:b] for a,b in zip(boundaries,boundaries[1:]))
        unknown = list(dict.fromkeys(unit for unit in units if unit not in self.known))
        if unknown:
            return None, unknown
        return PathResult(boundaries, units, float(distance[-1]),
                          math.fsum(self.known[u].quality for u in units)), []
