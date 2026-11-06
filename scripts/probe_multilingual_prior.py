"""Choose the fastest measured microbatch with the actual 77.6M FP16 model."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import statistics
import subprocess
import sys
import threading
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from char_diff.utils import ROOT,load_config,write_json


def worker(a):
    import numpy as np
    import torch
    from char_diff.bpe_packing import FrozenUnits
    from char_diff.flow import corrupt,loss_sum
    from char_diff.gpu_monitor import GPUMonitor
    from char_diff.model import Denoiser
    from char_diff.prior_stream import collate
    torch.set_num_threads(2); torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False; torch.manual_seed(1234)
    cfg=load_config(a.config); cfg['model']['grad_checkpoint']=a.checkpoint
    micro=a.worker; result=dict(micro_batch=micro,grad_checkpoint=a.checkpoint,status='started')
    monitor=None; thread=None; errors=[]
    try:
        model=Denoiser(cfg['model']).cuda().train(); ema=deepcopy(model).eval().requires_grad_(False)
        opt=torch.optim.AdamW(model.parameters(),lr=3e-4,betas=(.9,.95),foreach=True)
        scaler=torch.amp.GradScaler('cuda',init_scale=128)
        ae=FrozenUnits(ROOT/cfg['data']['ae_checkpoint'],'cuda',cfg['data']['ae_batch'])
        # 512 targets, maximal carry and 64 document controls: 1216 positions.
        rng=np.random.default_rng(712)
        parts=[dict(z=rng.normal(size=(577,48)).astype('f2'),prior=0,start=128,end=577,phase=3)]
        parts += [dict(z=rng.normal(size=(1,48)).astype('f2'),prior=0,start=0,end=1,phase=0) for _ in range(63)]
        batch=collate([parts]*micro,8,dict(mean=[0.]*48,scale=1.),'cuda')
        monitor=GPUMonitor(a.out.with_suffix('.gpu.jsonl'),interval=1).start()
        torch.cuda.reset_peak_memory_stats(); times=[]; losses=[]
        def frozen_peak():
            try:
                with torch.cuda.stream(torch.cuda.Stream()),torch.inference_mode():
                    until=time.perf_counter()+5
                    while time.perf_counter()<until:
                        ae.encode([b'The child walked into the house.']*cfg['data']['ae_batch'],score=True)
            except BaseException as exc:errors.append(exc)
        for step in range(4):
            tick=time.perf_counter(); opt.zero_grad(set_to_none=True)
            if step==0:thread=threading.Thread(target=frozen_peak);thread.start()
            x,t=corrupt(batch)
            with torch.autocast('cuda',dtype=torch.float16):
                pred=model(x,t,batch['roles'],batch['positions'],batch['lengths'],batch['allowed'])
                raw,n=loss_sum(pred,batch); loss=raw/(48*n)
            scaler.scale(loss).backward();scaler.unscale_(opt)
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            if not torch.isfinite(norm):raise FloatingPointError('Non-finite capacity gradient')
            scaler.step(opt);scaler.update()
            with torch.no_grad():torch._foreach_lerp_(list(ema.parameters()),list(model.parameters()),.001)
            if thread:thread.join();thread=None
            if errors:raise errors[0]
            torch.cuda.synchronize()
            if step:times.append(time.perf_counter()-tick);losses.append(float(loss))
        gpu=monitor.summary()
        result.update(status='ok',parameters=sum(p.numel() for p in model.parameters()),gpu=gpu,
                      median_seconds=statistics.median(times),seconds=times,losses=losses,
                      units_per_second=micro*512/statistics.median(times),physical_positions=batch['x'].shape[1])
    except BaseException as exc:
        result.update(status='oom' if 'out of memory' in str(exc).lower() else 'error',error=f'{type(exc).__name__}: {exc}')
    finally:
        if thread:thread.join(timeout=10)
        if monitor:monitor.close()
        write_json(a.out,result)
    return result


def choose(a):
    cfg=load_config(a.config); a.out.mkdir(parents=True,exist_ok=True)
    rows=[]
    def trial(micro,checkpoint):
        path=a.out/f'micro-{micro}-checkpoint-{int(checkpoint)}.json'
        command=[sys.executable,__file__,a.config,'--out',str(path),'--worker',str(micro)]
        if checkpoint:command+=['--checkpoint']
        subprocess.run(command,check=True,timeout=900)
        row=json.loads(path.read_text());rows.append(row)
        print(json.dumps(row),flush=True)
        if row['status']=='error':raise RuntimeError(row['error'])
        return row
    for micro in (4,8,12,16,24,32,48,64):
        row=trial(micro,False)
        if row['status']!='ok' or row['gpu']['peak_reserved_percent']>=93:break
    preceding=[r for r in rows[:-1] if r['status']=='ok']
    if preceding and micro-preceding[-1]['micro_batch']>1:
        trial((micro+preceding[-1]['micro_batch'])//2,False)
    eligible=[r for r in rows if r['status']=='ok' and r['gpu']['peak_reserved_percent']<=94
              and r['gpu'].get('nvidia_memory_peak_percent',0)<=96]
    first=max(eligible,key=lambda x:x['units_per_second']) if eligible else None
    # Compare recomputation only at useful larger microbatches.
    candidate=min(64,max(8,4*(first['micro_batch'] if first else 2)))
    row=trial(candidate,True)
    if row['status']=='ok' and row['gpu']['peak_reserved_percent']<80 and candidate<64:
        trial(min(64,candidate*2),True)
    elif row['status']=='oom' and candidate>4:trial(candidate//2,True)
    eligible=[r for r in rows if r['status']=='ok' and r['gpu']['peak_reserved_percent']<=94
              and r['gpu'].get('nvidia_memory_peak_percent',0)<=96]
    if not eligible:raise RuntimeError('Capacity probe has no admissible working point')
    selected=max(eligible,key=lambda x:x['units_per_second'])
    cfg['train']['micro_batch']=selected['micro_batch']
    cfg['model']['grad_checkpoint']=selected['grad_checkpoint']
    write_json(a.out/'production.json',cfg)
    write_json(a.out/'capacity.json',dict(selected=selected,trials=rows,
                criteria='highest measured throughput; reserved <=94%, device <=96%; concurrent AE peak included'))
    return selected


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('config'); p.add_argument('--out',type=Path,required=True)
    p.add_argument('--worker',type=int,default=0);p.add_argument('--checkpoint',action='store_true')
    a=p.parse_args();result=worker(a) if a.worker else choose(a)
    print(json.dumps(result),flush=True)
