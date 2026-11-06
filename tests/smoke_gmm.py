"""Two approved real-checkpoint warm starts, two FP16 updates each, both samplers."""
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from char_diff.flow import generate
from char_diff.model import Denoiser
from char_diff.training import train
from char_diff.utils import ROOT,load_config,write_json


if __name__=='__main__':
    report={'date':'2026-11-02','device':torch.cuda.get_device_name(),'torch':torch.__version__,'arms':{}}
    parents={'early':'weights_epoch_1.6.pt','late':'last.pt'}
    for arm,filename in parents.items():
        cfg=load_config(ROOT/f'configs/tinystories_gmm_{arm}.yaml')
        cfg['storage']['require_quota']=False
        cfg['train']['run_dir']=f'runs/gmm_{arm}_fp16_smoke'
        cfg['train']['log_every']=1
        source=ROOT/'runs/tinystories_x0_22m'/filename
        result=train(cfg,gmm_from=source,stop_after=2,device='cuda')
        state=torch.load(result['checkpoint'],map_location='cpu',weights_only=False)
        rows=[json.loads(s) for s in (ROOT/cfg['train']['run_dir']/'train.jsonl').read_text().splitlines()]
        assert sum(r['updated'] for r in rows)==2
        model=Denoiser(cfg['model']).cuda().eval();model.load_state_dict(state['ema'])
        for method in ['euler','gm_ode']:
            with torch.autocast('cuda',dtype=torch.float16):
                z=generate(model,torch.empty(0,48),16,steps=32,seed=42,sampler=method,substeps=4)
            assert z.shape==(16,48) and torch.isfinite(z).all()
        report['arms'][arm]={'source_step':state['lineage']['source_step'],'final_step':state['state']['step'],
                            'successful_updates':2,'skipped_updates':sum(not r['updated'] for r in rows),
                            'losses':[r['loss'] for r in rows],'learning_rates':[r['lr'] for r in rows],
                            'peak_cuda_bytes':max(r['peak_cuda_bytes'] for r in rows),
                            'checkpoint':result['checkpoint'],'samplers':['euler','gm_ode_4_substeps'],
                            'seconds':result['elapsed_seconds']}
        write_json(ROOT/'docs/experiments/gmm-switch-timing/evidence/fp16-smoke.json',report)
        del model,state
        torch.cuda.empty_cache()
    print(json.dumps(report),flush=True)
