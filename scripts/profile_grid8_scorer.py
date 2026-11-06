"""One resident FP32 batch: attribute time to actual GPU kernels and CPU work."""
import argparse,json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from char_diff.ae_scoring_runtime import OptimizedAENLLScorer
from char_diff.ae_nll import tensorize_compact
from char_diff.ae_quality import targets_from_lengths
from char_diff.vendor.bltz.config import Cfg
from char_diff.vendor.bltz.models.autoencoder import ByteStringAE

p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--out',type=Path,required=True)
a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True);torch.set_num_threads(4)
state=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
ae=ByteStringAE(Cfg(state['cfg']));ae.load_state_dict(state['model']);ae.requires_grad_(False);ae.eval()
scorer=OptimizedAENLLScorer(ae,device='cuda',batch_size=8192)
units=[b'Once upon a time there was a cat.'[:32]]*8192
scorer(units)
with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:
    for _ in range(3):scorer(units)
table=prof.key_averages().table(sort_by='self_cuda_time_total',row_limit=30)
(a.out/'profile.txt').write_text(table,encoding='utf8');print(table,flush=True)
