"""Affected scheduler, compiled tie handling and cache ownership checks."""
from pathlib import Path
import sys,tempfile,random
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from char_diff.ae_path_runtime import CompiledLazyGraph,FrequentScoreCache,solve_stream
from char_diff.ae_paths import CandidateScore,candidate_spans,shortest_path


if __name__=='__main__':
    rng=random.Random(42);docs=[];table={}
    for _ in range(24):
        raw=''.join(rng.choice(['a','b','苹','果','🙂']) for _ in range(20)).encode()
        docs.append({'raw':raw})
        for start,end in candidate_spans(raw):
            unit=raw[start:end]
            if unit not in table:table[unit]=CandidateScore(len(unit.decode())==1 or rng.random()>.25,rng.choice([0.,.25,1.]))
    docs+=docs[:6]
    calls=[]
    def scorer(units):calls.extend(units);return [table[u] for u in units]
    cache=FrequentScoreCache(1000,{'source':'fixture'})
    rows,stats=solve_stream(docs,scorer,active_documents=6,batch_size=32,score_cache=cache)
    for row in rows:
        raw=row['row']['raw'];reference=shortest_path(raw,{raw[s:e]:table[raw[s:e]] for s,e in candidate_spans(raw)},quality_weight=10)
        assert row['path'].boundaries==reference.boundaries
    assert cache.hits>0 and len(cache.hot)+len(cache.probation)<=1000
    with tempfile.TemporaryDirectory(dir=Path.cwd()/'runs',prefix='cache_check_') as d:
        path=Path(d)/'cache.bin';cache.save(path);restored=FrequentScoreCache(1000,{'source':'fixture'});restored.load(path)
        assert list(cache.hot.items())==list(restored.hot.items())
        try:FrequentScoreCache(1000,{'source':'wrong'}).load(path)
        except ValueError:pass
        else:raise AssertionError('Wrong cache scope accepted')
    print({'exact_paths':len(rows),'scored':len(calls),'cache_hits':cache.hits,'cache_roundtrip':True})
