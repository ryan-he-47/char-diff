"""Affected-path checks: exact lazy optimum, ties, UTF-8, repeated strings, no path."""
import math
from pathlib import Path
import random
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from char_diff.ae_lazy_paths import LazyNLLGraph
from char_diff.ae_paths import candidate_spans, CandidateScore, shortest_path, NoFeasiblePath


def check(raw, scores, weight, max_bytes):
    graph=LazyNLLGraph(raw,weight=weight,max_bytes=max_bytes)
    for _ in range(len(scores)+1):
        result,pending=graph.optimistic_path()
        if result is not None:break
        graph.update({u:scores[u] for u in pending})
    else:raise AssertionError('Lazy solver did not finish')
    reference=shortest_path(raw,scores,quality_weight=weight,max_bytes=max_bytes)
    assert result.boundaries==reference.boundaries,(raw,weight,result,reference)
    assert math.isclose(result.cost,reference.cost,rel_tol=1e-12,abs_tol=1e-12)


if __name__=='__main__':
    rng=random.Random(20261105)
    for number in range(60):
        raw=''.join(rng.choice(['a','b','苹','果','é','🙂']) for _ in range(15)).encode()
        units={raw[s:e] for s,e in candidate_spans(raw,max_bytes=8)}
        scores={u:CandidateScore(len(u.decode())==1 or rng.random()>.3,
                                rng.choice([0.,0.,.125,.25,1.])) for u in units}
        check(raw,scores,rng.choice([0.,1.,10.]),8)
    check(b'',{},10.,32)
    graph=LazyNLLGraph('苹果'.encode(),max_bytes=8)
    graph.update({u:CandidateScore(False) for u in graph.occurrences})
    try:graph.optimistic_path()
    except NoFeasiblePath:pass
    else:raise AssertionError('Expected infeasible graph')
    print('60 exact-optimum/tie cases, empty input and infeasible graph passed')
