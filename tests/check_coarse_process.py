"""Process lifecycle, exact graph paths, compact messages and GPU error exit."""
import os
os.environ['OMP_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['MKL_NUM_THREADS']='1'
from pathlib import Path
import sys,random,json,tempfile,signal
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from char_diff.ae_paths import CandidateScore
from char_diff.ae_path_runtime import FrequentScoreCache
from char_diff.coarse_paths import solve_coarse_batch
from char_diff.coarse_process import ProcessCoarseSolver,pack_candidates,unpack_candidates

if __name__=='__main__':
    rng=random.Random(42)
    docs=[{'raw':''.join(rng.choice(['a','\0','果','苹','🙂']) for _ in range(50)).encode()} for _ in range(24)]
    docs+=docs[:8]
    def score(units):return [CandidateScore(len(u.decode())==1 or sum(u)%7!=0,(sum(u)%5)/16) for u in units]
    assert unpack_candidates(pack_candidates([b'\0',b'abc','果🙂'.encode()]))==[b'\0',b'abc','果🙂'.encode()]
    reference,_=solve_coarse_batch(docs,score,lazy=False)
    solver=ProcessCoarseSolver(2)
    cache=FrequentScoreCache(20000)
    actual,stats=solver.solve(docs,score,group_documents=3,score_cache=cache)
    assert [r['index'] for r in actual]==list(range(len(docs)))
    assert [r['path'] for r in actual]==[r['path'] for r in reference]
    if hasattr(signal,'SIGUSR1'):
        for process in solver.processes:os.kill(process.pid,signal.SIGUSR1)
    with tempfile.TemporaryDirectory(dir='runs') as folder:
        cache.save(Path(folder)/'scores.bin');restored=FrequentScoreCache(20000);restored.load(Path(folder)/'scores.bin')
        again,_=solver.solve(docs[:2],score,group_documents=1,score_cache=restored)
    assert [r['path'] for r in again]==[r['path'] for r in reference[:2]]
    def broken(units):raise RuntimeError('injected GPU failure')
    try:solver.solve(docs,broken,group_documents=3)
    except RuntimeError as exc:assert str(exc)=='injected GPU failure'
    else:raise AssertionError('Missing propagated failure')
    assert all(not p.is_alive() for p in solver.processes)
    print(json.dumps({'same_paths':len(actual),'pool_reuse':True,'cache_reload':True,
                      'worker_usr1':hasattr(signal,'SIGUSR1'),'failure_cleanup':True,'stats':stats}))
