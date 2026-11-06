"""Bounded coarse candidate solving, with a shared frequent-fragment cache."""
from __future__ import annotations

import time

from .ae_path_runtime import CompiledLazyGraph, FrequentScoreCache, solve_stream
from .ae_paths import boundary_offsets
from .path_proposals import proposed_boundaries


def solve_coarse_batch(documents, scorer, *, mode='grid8', tokenizer=None,
                       weight=10., score_cache=None, batch_size=2048, lazy=True):
    """Solve complete documents; caller bounds the document/byte batch size.

    Adjacent proposed spans are scored first. Failed spans receive all their
    interior character boundaries. The final objective is exact on this graph.
    Each candidate keeps its original bytes, including tokenizer-normalized text.
    """
    if batch_size < 1 or not documents:
        raise ValueError('Expected a positive batch size and nonempty documents')
    started=time.perf_counter()
    cache=score_cache if score_cache is not None else FrequentScoreCache(0)
    scored=0;score_seconds=0.;scores={}

    def collect(units):
        nonlocal scored,score_seconds
        missing=[]
        for unit in dict.fromkeys(units):
            if unit in scores:
                continue
            value=cache.get(unit)
            if value is None:missing.append(unit)
            else:scores[unit]=value
        if missing:
            # Let the scorer form full length buckets over the bounded group.
            tick=time.perf_counter();values=scorer(missing)
            score_seconds+=time.perf_counter()-tick;scored+=len(missing)
            if len(values)!=len(missing):raise ValueError('Incorrect scorer result length')
            for unit,value in zip(missing,values):
                scores[unit]=value;cache.put(unit,value)

    proposed=[proposed_boundaries(d['raw'],mode,tokenizer=tokenizer) for d in documents]
    collect(d['raw'][a:b] for d,nodes in zip(documents,proposed) for a,b in zip(nodes,nodes[1:]))
    prepared=[];fallbacks=0
    for document,nodes in zip(documents,proposed):
        raw=document['raw'];full=boundary_offsets(raw);expanded=set(nodes)
        for a,b in zip(nodes,nodes[1:]):
            if not scores[raw[a:b]].feasible:
                expanded.update(x for x in full if a<x<b);fallbacks+=1
        prepared.append({**document,'offsets':tuple(sorted(expanded)), 'initial_scores':scores})
    if lazy:
        rows,timing=solve_stream(prepared,scorer,weight=weight,
                                active_documents=len(prepared),batch_size=batch_size,
                                score_cache=cache)
        scored+=timing['scored_candidates'];score_seconds+=timing['score_wall_seconds']
    else:
        graphs=[CompiledLazyGraph(d['raw'],weight=weight,offsets=d['offsets']) for d in prepared]
        collect(unit for graph in graphs for unit in graph.occurrences)
        rows=[]
        for index,(document,graph) in enumerate(zip(prepared,graphs)):
            graph.update({u:scores[u] for u in graph.occurrences});path,pending=graph.optimistic_path()
            assert not pending
            rows.append({'index':index,'row':document,'path':path,'rounds':1})
    for row in rows:
        row['row']={k:v for k,v in row['row'].items() if k!='initial_scores'}
    return rows,{'seconds':time.perf_counter()-started,'scored_candidates':scored,
                 'score_wall_seconds':score_seconds,'fallback_intervals':fallbacks}
