"""Packed CUDA input/output compared with the existing FP32 batch scorer."""
import argparse,json,random,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from char_diff.ae_scoring_runtime import OptimizedAENLLScorer
from char_diff.ae_scoring_array import PackedAENLLScorer,ConcurrentPackedScorer
from char_diff.ae_scoring_pipeline import ConcurrentAENLLScorer
from char_diff.coarse_process import pack_candidates
from char_diff.vendor.bltz.config import Cfg
from char_diff.vendor.bltz.models.autoencoder import ByteStringAE

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);a=p.parse_args()
    state=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
    ae=ByteStringAE(Cfg(state['cfg']));ae.load_state_dict(state['model']);ae.requires_grad_(False)
    rng=random.Random(123);units=[]
    for _ in range(1600):
        raw=''.join(rng.choice(['a','\0','果','猫','🙂','ع']) for _ in range(rng.randint(1,12))).encode()
        if len(raw)<=32:units.append(raw)
    reference=OptimizedAENLLScorer(ae,device='cuda',batch_size=128)(units)
    expected=np.asarray([(v.feasible,v.quality) for v in reference],np.float64)
    packed=pack_candidates(units);actual=PackedAENLLScorer(ae,device='cuda',batch_size=128).score_packed(packed)
    assert np.array_equal(expected,actual)
    reference_parallel=ConcurrentAENLLScorer(ae,batch_size=128,streams=2);reference_parallel.chunk=256
    try:parallel_values=reference_parallel(units)
    finally:reference_parallel.close()
    parallel_expected=np.asarray([(v.feasible,v.quality) for v in parallel_values])
    concurrent=ConcurrentPackedScorer(ae,batch_size=128,streams=2);concurrent.chunk=256
    try:other=concurrent.score_packed(packed)
    finally:concurrent.close()
    error=float(np.max(abs(other[:,1]-parallel_expected[:,1])))
    assert np.array_equal(other,parallel_expected)
    print(json.dumps({'exact_single_stream_scores':len(units),'exact_two_stream_scores':len(units),'same_chunk_max_nll_delta':error,
                      'single_to_two_stream_batch_shape_delta':float(np.max(abs(other[:,1]-expected[:,1])))}))
