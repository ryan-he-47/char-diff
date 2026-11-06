"""Exact two-queue behavior and bidirectional compatibility with saved caches."""
import sys,random,tempfile,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from char_diff.ae_paths import CandidateScore
from char_diff.ae_path_runtime import FrequentScoreCache
from char_diff.compiled_score_cache import CompiledScoreCache
from char_diff.coarse_process import pack_candidates

if __name__=='__main__':
    rng=random.Random(17);units=[f'item{i}'.encode() for i in range(60)]+[b'\0','果🙂'.encode(),b'a'*32]
    counts=0
    with tempfile.TemporaryDirectory(dir='runs') as folder:
        root=Path(folder)
        for capacity in (0,1,2,7,100):
            reference=FrequentScoreCache(capacity,{'test':1});actual=CompiledScoreCache(capacity,{'test':1})
            for step in range(80):
                batch=[rng.choice(units) for _ in range(rng.randint(1,25))]
                refs=[reference.get(u) for u in batch];values,missing=actual.query(pack_candidates(batch))
                expected_missing=[i for i,r in enumerate(refs) if r is None]
                assert missing.tolist()==expected_missing
                for i,r in enumerate(refs):
                    if r is not None:assert tuple(values[i])==(r.feasible,r.quality)
                scores=np.asarray([(sum(batch[i])%3!=0,(sum(batch[i])%101)/32) for i in missing],np.float64).reshape(-1,2)
                actual.update(pack_candidates([batch[i] for i in missing]),scores)
                for i,(ok,q) in zip(missing,scores):reference.put(batch[i],CandidateScore(bool(ok),q))
                assert actual.statistics()==reference.statistics();counts+=len(batch)
            reference.save(root/'reference.bin');actual.save(root/'actual.bin')
            assert (root/'reference.bin').read_bytes()==(root/'actual.bin').read_bytes()
            loaded=CompiledScoreCache(capacity,{'test':1});loaded.load(root/'reference.bin');loaded.save(root/'roundtrip.bin')
            assert (root/'roundtrip.bin').read_bytes()==(root/'reference.bin').read_bytes()
            small=CompiledScoreCache(3,{'test':1});small.load(root/'actual.bin')
            small_ref=FrequentScoreCache(3,{'test':1});small_ref.load(root/'actual.bin')
            small.save(root/'small.bin');small_ref.save(root/'small_ref.bin')
            assert (root/'small.bin').read_bytes()==(root/'small_ref.bin').read_bytes()
    print(json.dumps({'exact_queries':counts,'capacities':[0,1,2,7,100],'statistics_and_lru':True,'binary_roundtrip':True}))
