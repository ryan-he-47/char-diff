"""Compare the double-buffered CUDA scorer to the same FP32 batch protocol."""
from pathlib import Path
import argparse,json,random,sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from char_diff.ae_scoring_runtime import OptimizedAENLLScorer
from char_diff.ae_scoring_pipeline import PipelinedAENLLScorer
from char_diff.vendor.bltz.config import Cfg
from char_diff.vendor.bltz.models.autoencoder import ByteStringAE

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);a=p.parse_args()
    state=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
    ae=ByteStringAE(Cfg(state['cfg']));ae.load_state_dict(state['model']);ae.requires_grad_(False)
    rng=random.Random(123);units=[]
    for _ in range(1500):
        raw=''.join(rng.choice(['a',' ','果','猫','🙂','ع']) for _ in range(rng.randint(1,12))).encode()
        if len(raw)<=32:units.append(raw)
    reference=OptimizedAENLLScorer(ae,device='cuda',batch_size=128)(units)
    actual=PipelinedAENLLScorer(ae,device='cuda',batch_size=128)(units)
    assert reference==actual
    print(json.dumps({'exact_scores':len(units),'mixed_utf8_buckets':True,'cuda_double_buffer':True}))
