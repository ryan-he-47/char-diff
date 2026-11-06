"""Pipeline order, duplicate scoring, UTF-8 fallback and failure propagation."""
from pathlib import Path
import sys,random
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from char_diff.ae_paths import CandidateScore,NoFeasiblePath
from char_diff.ae_path_runtime import FrequentScoreCache
from char_diff.coarse_paths import solve_coarse_batch
from char_diff.coarse_pipeline import solve_coarse_pipeline

if __name__=='__main__':
    rng=random.Random(42)
    docs=[{'raw':''.join(rng.choice(['a','b','果','苹','🙂']) for _ in range(50)).encode()} for _ in range(24)]
    docs+=docs[:8]
    def score(units):return [CandidateScore(len(u.decode())==1 or sum(u)%7!=0,(sum(u)%5)/16) for u in units]
    reference,_=solve_coarse_batch(docs,score,lazy=False)
    actual,stats=solve_coarse_pipeline(docs,score,workers=4,group_documents=3,score_cache=FrequentScoreCache(20000))
    assert [r['index'] for r in actual]==list(range(len(docs)))
    assert [r['path'] for r in actual]==[r['path'] for r in reference]
    def broken(units):raise RuntimeError('injected GPU failure')
    try:solve_coarse_pipeline(docs,broken,workers=4,group_documents=3)
    except RuntimeError as exc:assert str(exc)=='injected GPU failure'
    else:raise AssertionError('Missing propagated failure')
    print({'same_paths':len(actual),'failure_propagation':True,'stats':stats})
