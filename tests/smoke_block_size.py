"""Two real-source FP16 updates per block size; temporary full checkpoints are discarded."""
import json
import sys
import tempfile
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from char_diff.flow import generate
from char_diff.model import Denoiser
from char_diff.training import train
from char_diff.utils import ROOT,load_config,write_json


if __name__=='__main__':
    report={'device':torch.cuda.get_device_name(),'arms':{}}
    for block in [1,4]:
        cfg=load_config(ROOT/f'configs/tinystories_x0_block{block}.yaml')
        cfg['storage']['require_quota']=False;cfg['train']['log_every']=1
        with tempfile.TemporaryDirectory(prefix=f'block{block}_fp16_',dir=ROOT/'runs') as temporary:
            folder=Path(temporary).resolve();folder.relative_to((ROOT/'runs').resolve())
            cfg['train']['run_dir']=str(folder)
            result=train(cfg,block_from=ROOT/'runs/tinystories_x0_22m/weights_epoch_1.6.pt',stop_after=2,device='cuda')
            state=torch.load(result['checkpoint'],map_location='cpu',weights_only=False)
            rows=[json.loads(s) for s in (folder/'train.jsonl').read_text().splitlines()]
            assert len(rows)==2 and all(r['updated'] for r in rows)
            assert state['state']['step']==20615 and state['sampler']['epoch']==1 and state['sampler']['cursor']==496054
            model=Denoiser(cfg['model']).cuda().eval();model.load_state_dict(state['ema'])
            with torch.autocast('cuda',dtype=torch.float16):z=generate(model,torch.empty(0,48),block*2,steps=2,seed=42)
            assert z.shape==(block*2,48) and torch.isfinite(z).all()
            report['arms'][str(block)]={'start_step':20613,'end_step':20615,'updates':2,'skips':0,
                                        'parameters':sum(p.numel() for p in model.parameters()),
                                        'losses':[r['loss'] for r in rows],'grad_norms':[r['grad_norm'] for r in rows],
                                        'learning_rates':[r['lr'] for r in rows],
                                        'peak_cuda_bytes':max(r['peak_cuda_bytes'] for r in rows),
                                        'seconds':result['elapsed_seconds'],'sampler':state['sampler'],
                                        'native_block_generation_finite':True}
            write_json(ROOT/'docs/experiments/block-size/evidence/fp16-smoke.json',report)
            del model,state,z
            torch.cuda.empty_cache()
    print(json.dumps(report),flush=True)
