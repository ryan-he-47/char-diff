"""Audit frozen inference refactors on identical candidates and GPU stages."""
import argparse,hashlib,json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from char_diff.ae_nll import AENLLScorer
from char_diff.ae_scoring_runtime import OptimizedAENLLScorer
from char_diff.ae_paths import candidate_spans
from char_diff.vendor.bltz.config import Cfg
from char_diff.vendor.bltz.models.autoencoder import ByteStringAE
from char_diff.utils import write_json,sha256


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--documents',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--batch-size',type=int,default=2048)
    p.add_argument('--min-bytes',type=int,default=1)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True);torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    rows=[json.loads(s) for s in a.documents.read_text(encoding='utf8').splitlines() if s]
    units={}
    for row in rows:
        raw=bytes.fromhex(row['source_hex'])
        candidates={raw[s:e] for s,e in candidate_spans(raw)}
        units.update(dict.fromkeys(sorted(candidates,key=lambda u:hashlib.sha256(u).digest())[:512]))
    units=[u for u in units if len(u)>=a.min_bytes]
    state=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
    ae=ByteStringAE(Cfg(state['cfg']));ae.load_state_dict(state['model']);ae.requires_grad_(False)
    scorers={'original':AENLLScorer(ae,device='cuda',batch_size=a.batch_size),
             'factored':OptimizedAENLLScorer(ae,device='cuda',batch_size=a.batch_size,bucketed=False),
             'bucketed_factored':OptimizedAENLLScorer(ae,device='cuda',batch_size=a.batch_size)}
    result={'checkpoint_sha256':sha256(a.checkpoint),'input_sha256':sha256(a.documents),'gpu':torch.cuda.get_device_name(),
            'candidates':len(units),'min_bytes':a.min_bytes,'mean_bytes':float(np.mean(list(map(len,units)))),'batch_size':a.batch_size,'variants':{}}
    reference=None
    for name,scorer in scorers.items():
        scorer(units[:a.batch_size]);torch.cuda.synchronize();tick=time.perf_counter()
        values=scorer(units);torch.cuda.synchronize();seconds=time.perf_counter()-tick
        if reference is None:reference=values
        errors=np.array([abs(x.quality-y.quality) for x,y in zip(reference,values)])
        data={'seconds':seconds,'candidates_per_second':len(units)/seconds,
              'feasibility_flips':sum(x.feasible!=y.feasible for x,y in zip(reference,values)),
              'mean_abs_nll_error':float(errors.mean()),'max_abs_nll_error':float(errors.max())}
        result['variants'][name]=data;write_json(a.out/'results.json',result);print(json.dumps({name:data}),flush=True)
