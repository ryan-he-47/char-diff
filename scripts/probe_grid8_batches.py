"""Measure real coarse-candidate FP32 batches and GPU active time."""
import argparse,json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from char_diff.ae_scoring_runtime import OptimizedAENLLScorer
from char_diff.ae_nll import tensorize_compact
from char_diff.ae_quality import targets_from_lengths
from char_diff.path_proposals import proposed_boundaries
from char_diff.vendor.bltz.config import Cfg
from char_diff.vendor.bltz.models.autoencoder import ByteStringAE
from char_diff.utils import write_json


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--reference',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--batches',type=int,nargs='+',default=[2048,8192,32768,49152,65536])
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True);torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    state=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
    ae=ByteStringAE(Cfg(state['cfg']));ae.load_state_dict(state['model']);ae.requires_grad_(False)
    ref=json.loads(a.reference.read_text(encoding='utf8'))['variants']['8']['rows']
    pool={}
    for row in ref:
        raw=bytes.fromhex(row['source_hex']);nodes=proposed_boundaries(raw,'grid8')
        for i,start in enumerate(nodes[:-1]):
            for end in nodes[i+1:]:
                if end-start>32:break
                pool[raw[start:end]]=None
    # Fixed 32B candidates expose the largest real inference shape.
    long=[u for u in pool if len(u)==32];units=(long*((131072+len(long)-1)//len(long)))[:131072]
    result={'device':torch.cuda.get_device_name(),'total_bytes':torch.cuda.get_device_properties(0).total_memory,
            'shape':'131072 real 32B candidates, repeated only to fill each tested batch','trials':[]}
    baseline=None
    for batch in a.batches:
        torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
        scorer=OptimizedAENLLScorer(ae,device='cuda',batch_size=batch)
        try:
            scorer(units[:batch]);torch.cuda.synchronize();tick=time.perf_counter()
            values=scorer(units);torch.cuda.synchronize();elapsed=time.perf_counter()-tick
            if baseline is None:baseline=values
            data={'batch':batch,'seconds':elapsed,'candidates_per_second':len(units)/elapsed,
                  'peak_allocated':torch.cuda.max_memory_allocated(),'peak_reserved':torch.cuda.max_memory_reserved(),
                  'feasibility_flips':sum(x.feasible!=y.feasible for x,y in zip(values,baseline)),
                  'max_nll_error':max(abs(x.quality-y.quality) for x,y in zip(values,baseline))}
        except (torch.cuda.OutOfMemoryError,RuntimeError) as exc:
            data={'batch':batch,'error':str(exc)[:500]};torch.cuda.empty_cache()
        result['trials'].append(data);write_json(a.out/'results.json',result);print(json.dumps(data),flush=True)
