"""Measure AE path-scoring throughput and FP16 scoring drift; no training."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from char_diff.ae_nll import AENLLScorer
from char_diff.ae_paths import candidate_spans
from char_diff.utils import write_json, sha256
from char_diff.vendor.bltz.config import Cfg
from char_diff.vendor.bltz.models.autoencoder import ByteStringAE


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--reference-paths',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--batches',type=int,nargs='+',default=[512,2048,8192,16384,32768])
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True);torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    state=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
    ae=ByteStringAE(Cfg(state['cfg']));ae.load_state_dict(state['model']);ae.requires_grad_(False)
    rows=[json.loads(s) for s in a.reference_paths.read_text(encoding='utf8').splitlines() if s]
    pool={}
    for row in rows:
        raw=bytes.fromhex(row['source_hex'])
        candidates={raw[s:e] for s,e in candidate_spans(raw)}
        selected=sorted(candidates,key=lambda u:hashlib.sha256(u).digest())[:1024]
        pool.update(dict.fromkeys(selected))
    pool=list(pool);repeat=max(a.batches)*2
    units=(pool*((repeat+len(pool)-1)//len(pool)))[:repeat]
    result={'checkpoint_sha256':sha256(a.checkpoint),'reference_sha256':sha256(a.reference_paths),
            'gpu':torch.cuda.get_device_name(),'total_memory_bytes':torch.cuda.get_device_properties(0).total_memory,
            'torch_version':torch.__version__,'source_candidates':len(pool),'trials':[],
            'purpose':'Inference throughput sizing; fp16 is diagnostic, official path precision remains fp32'}
    for precision in ['fp32','fp16']:
        for batch in a.batches:
            torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
            scorer=AENLLScorer(ae,device='cuda',batch_size=batch,precision=precision)
            try:
                scorer(units[:batch]);torch.cuda.synchronize()
                tick=time.perf_counter();scores=scorer(units[:2*batch]);torch.cuda.synchronize()
                seconds=time.perf_counter()-tick
                trial={'precision':precision,'batch':batch,'candidates':2*batch,'seconds':seconds,
                       'candidates_per_second':2*batch/seconds,'finite':all(np.isfinite(x.quality) for x in scores),
                       'allocated_bytes':torch.cuda.max_memory_allocated(),'reserved_bytes':torch.cuda.max_memory_reserved()}
                del scores
            except torch.cuda.OutOfMemoryError:
                trial={'precision':precision,'batch':batch,'oom':True}
                torch.cuda.empty_cache()
            result['trials'].append(trial);write_json(a.out/'results.json',result)
            print(json.dumps(trial),flush=True)
            if trial.get('oom'):break
    panel=pool[:8192]
    fp32=AENLLScorer(ae,device='cuda',batch_size=512,precision='fp32')(panel)
    fp16=AENLLScorer(ae,device='cuda',batch_size=512,precision='fp16')(panel)
    error=np.array([abs(a.quality-b.quality) for a,b in zip(fp32,fp16)])
    result['precision_drift']={'candidates':len(panel),'feasibility_flips':sum(a.feasible!=b.feasible for a,b in zip(fp32,fp16)),
                               'fp32_feasible':sum(a.feasible for a in fp32),'fp16_feasible':sum(a.feasible for a in fp16),
                               'mean_abs_nll_error':float(error.mean()),'max_abs_nll_error':float(error.max())}
    write_json(a.out/'results.json',result);print(json.dumps(result['precision_drift']),flush=True)


if __name__=='__main__':main()
