"""Full-corpus x0 training with transactional streamed data and mixed blocks."""
from __future__ import annotations

from collections import Counter, deque
from concurrent.futures import ThreadPoolExecutor
import copy
import json
import math
import random
import signal
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

from .bpe_packing import FrozenUnits,prepare_assets
from .flow import corrupt,loss_sum
from .gpu_monitor import GPUMonitor
from .model import Denoiser
from .prior_stream import GroupProvider,UnitStream,PrefetchedUnits,collate
from .storage import StorageGuard
from .training import atomic_checkpoint,rng_state,restore_rng
from .utils import ROOT,sha256,write_json


class BackgroundCheckpoints:
    """Snapshot at an update boundary, then overlap GPFS writes with training."""
    def __init__(self):
        self.executor=ThreadPoolExecutor(1,thread_name_prefix='checkpoint')
        self.future=None

    @staticmethod
    def cpu_snapshot(value):
        if isinstance(value,torch.Tensor):return value.detach().to('cpu',copy=True)
        if isinstance(value,dict):return {k:BackgroundCheckpoints.cpu_snapshot(v) for k,v in value.items()}
        if isinstance(value,list):return [BackgroundCheckpoints.cpu_snapshot(v) for v in value]
        if isinstance(value,tuple):return tuple(BackgroundCheckpoints.cpu_snapshot(v) for v in value)
        return copy.deepcopy(value)

    def check(self):
        if self.future is not None and self.future.done():self.future.result()

    def submit(self,path,payload,guard,estimate,rotate=False):
        # Bound RAM to one in-flight snapshot and preserve last/previous order.
        if self.future is not None:self.future.result()
        snapshot=self.cpu_snapshot(payload)
        self.future=self.executor.submit(atomic_checkpoint,path,snapshot,guard,estimate,rotate)
        return self.future

    def close(self):
        self.executor.shutdown(wait=True)
        if self.future is not None:self.future.result()


def setup_provider(cfg,device,root=ROOT):
    dc=cfg['data']; root=Path(root)
    if sha256(root/dc['ae_checkpoint'])!=dc['ae_sha256']:raise ValueError('Frozen AE changed')
    encoder=FrozenUnits(root/dc['ae_checkpoint'],device,dc['ae_batch'])
    if dc['mode']=='bpe':
        prepare_assets(root/dc['tokenizer'],root/dc['ae_checkpoint'],root/dc['vocab_scores'],
                       root/dc['bpe_assets'],encoder,dc['nll_threshold'])
    return GroupProvider(cfg,encoder,root)


def byte_lr(end,budget,tc):
    warm=budget*tc['warmup_fraction']; decay=budget*(1-tc['decay_fraction'])
    if end<warm:return tc['lr']*max(1,end)/max(1,warm)
    if end>=decay:
        fraction=min(1,max(0,(end-decay)/max(1,budget-decay)))
        return tc['lr']+(tc['min_lr']-tc['lr'])*fraction
    return tc['lr']


def train(cfg,*,resume=None,stop_after=0,device='cuda',root=ROOT,monitor=True):
    root=Path(root); device=torch.device(device); tc=cfg['train']; dc=cfg['data']
    torch.set_num_threads(2); torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    seed=tc['seed']; random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if device.type=='cuda':torch.cuda.manual_seed_all(seed)
    run=(root/tc['run_dir']).resolve(); run.relative_to(root.resolve()); run.mkdir(parents=True,exist_ok=True)
    if not resume and any((run/p).exists() for p in ('config.json','last.pt','train.jsonl')):
        raise FileExistsError('Choose explicit resume for an existing prior run')
    guard=StorageGuard(root,limit_gib=cfg['storage']['project_limit_gib'],
                       reserve_gib=cfg['storage']['reserve_gib'],require_quota=cfg['storage']['require_quota'])
    guard.check()
    provider=setup_provider(cfg,device,root)
    normalizer=provider.normalizer(run/'normalizer.json')
    source=UnitStream(provider,tc['block_sizes'])
    model=Denoiser(cfg['model']).to(device)
    ema=copy.deepcopy(model).eval().requires_grad_(False)
    optimizer=torch.optim.AdamW(model.parameters(),lr=tc['lr'],betas=tuple(tc['betas']),eps=tc['eps'],
                               weight_decay=tc['weight_decay'],foreach=True)
    amp=device.type=='cuda' and tc['precision']=='fp16'
    scaler=torch.amp.GradScaler('cuda',enabled=amp,init_scale=1024,growth_interval=500)
    budget=int(provider.raw.spec['raw_bytes'])
    state=dict(step=0,attempts=0,seen_bytes=0,seen_units=0,skipped_attempts=0,
               attempted_bytes=0,saved_milestones=[],bytes_by_language={},units_by_block={})
    norms=deque(maxlen=100)
    if resume:
        saved=torch.load(resume,map_location='cpu',weights_only=False)
        for name in ('data','model','train'):
            if saved['config'][name]!=cfg[name]:raise ValueError('Resume configuration changed: '+name)
        if saved['cache_fingerprint']!=provider.fingerprint:raise ValueError('Resume corpus differs')
        model.load_state_dict(saved['model']); ema.load_state_dict(saved['ema'])
        optimizer.load_state_dict(saved['optimizer']); scaler.load_state_dict(saved['scaler'])
        source.load_state_dict(saved['sampler']); state=saved['state']; norms.extend(saved['norms'])
        if saved['normalizer']!=normalizer:raise ValueError('Resume normalizer changed')
        restore_rng(saved['rng']); del saved
    stream=PrefetchedUnits(source,tc['batch_windows'],dc['prefetch'])
    parameters=sum(p.numel() for p in model.parameters())
    code=Path(__file__).resolve().parents[1]
    source_sha={name:sha256(code/name) for name in ('char_diff/prior_training.py','char_diff/prior_stream.py',
                'char_diff/bpe_packing.py','char_diff/model.py','char_diff/flow.py','scripts/train_multilingual.py')}
    write_json(run/'config.json',cfg)
    stopped={'value':False}
    def request_stop(*args):stopped['value']=True
    signals=[signal.SIGTERM,signal.SIGINT]
    if hasattr(signal,'SIGUSR1'):signals.append(signal.SIGUSR1)
    old={s:signal.signal(s,request_stop) for s in signals}
    observer=None
    if device.type=='cuda':
        torch.cuda.reset_peak_memory_stats()
        if monitor:observer=GPUMonitor(run/'gpu.jsonl',interval=1).start()
    writer=BackgroundCheckpoints()
    last_saved=-1; started=time.perf_counter(); start_bytes=state['seen_bytes']; start_step=state['step']
    estimate=int(parameters*17+16*1024**2)

    def payload():
        return dict(format='multilingual-prior-v1',config=cfg,model=model.state_dict(),ema=ema.state_dict(),
            optimizer=optimizer.state_dict(),scaler=scaler.state_dict(),normalizer=normalizer,
            cache_fingerprint=provider.fingerprint,data_protocol=provider.protocol,
            codec=dict(ae_sha256=dc['ae_sha256'],checkpoint=dc['ae_checkpoint']),
            state=copy.deepcopy(state),sampler=stream.state_dict(),rng=rng_state(),norms=list(norms),
            budget_bytes=budget,source_sha256=source_sha)

    def save(force=False):
        nonlocal last_saved
        if force or state['attempts']!=last_saved:
            writer.submit(run/'last.pt',payload(),guard,estimate,rotate=True)
            last_saved=state['attempts']

    initial=dict(event='start',parameters=parameters,budget_bytes=budget,step=state['step'],
        seen_bytes=state['seen_bytes'],mode=dc['mode'],micro_batch=tc['micro_batch'],
        batch_windows=tc['batch_windows'],block_sizes=tc['block_sizes'],device=str(device))
    print(json.dumps(initial),flush=True); write_json(run/'status.json',initial)
    log=(run/'train.jsonl').open('a',encoding='utf8',buffering=1)
    failure=None; invalid=0; safe_model=True
    try:
        while state['seen_bytes']<budget:
            writer.check()
            tick=time.perf_counter(); batch_data=stream.next(); ready=time.perf_counter()
            if batch_data is None:
                raise ValueError('Raw stream ended before the exact byte budget')
            count=batch_data['units']; target_bytes=batch_data['raw_bytes']; block=batch_data['block']
            if state['seen_bytes']+target_bytes>budget:raise ValueError('Raw stream exceeded one pass')
            # Retry the same raw targets after numerical skips, preserving exact coverage.
            while True:
                step_start=time.perf_counter()
                lr=byte_lr(state['seen_bytes']+target_bytes,budget,tc)
                for group in optimizer.param_groups:group['lr']=lr
                optimizer.zero_grad(set_to_none=True); total_loss=0.; physical=0
                model.train()
                windows=batch_data['windows']
                for start in range(0,len(windows),tc['micro_batch']):
                    batch=collate(windows[start:start+tc['micro_batch']],block,normalizer,device)
                    x,t=corrupt(batch)
                    with torch.autocast(device_type=device.type,dtype=torch.float16,enabled=amp):
                        estimate_z=model(x,t,batch['roles'],batch['positions'],batch['lengths'],batch['allowed'])
                        raw,_=loss_sum(estimate_z,batch)
                        loss=raw/(48*count)
                    scaler.scale(loss).backward(); total_loss+=float(loss.detach())
                    physical+=batch['x'].shape[0]*batch['x'].shape[1]
                    del batch,x,t,estimate_z,raw,loss
                scaler.unscale_(optimizer)
                norm=float(torch.nn.utils.clip_grad_norm_(model.parameters(),tc['clip']))
                good=math.isfinite(norm) and math.isfinite(total_loss) and norm<=max(2000,10*float(np.median(norms)) if norms else 2000)
                state['attempts']+=1; state['attempted_bytes']+=target_bytes
                if good:
                    safe_model=False
                    scaler.step(optimizer); scaler.update(); state['step']+=1
                    state['seen_bytes']+=target_bytes; state['seen_units']+=count; norms.append(norm); invalid=0
                    state['bytes_by_language']=dict(Counter(state['bytes_by_language'])+Counter(batch_data['languages']))
                    key=str(block); state['units_by_block'][key]=state['units_by_block'].get(key,0)+count
                    with torch.no_grad():
                        torch._foreach_lerp_(list(ema.parameters()),list(model.parameters()),1-tc['ema'])
                    stream.commit(batch_data)
                    safe_model=True
                else:
                    scaler.update(); state['skipped_attempts']+=1; invalid+=1
                    # Oversized finite gradients also reduce the FP16 scale on retry.
                    if amp and math.isfinite(norm):scaler.update(new_scale=max(1.,scaler.get_scale()/2))
                if device.type=='cuda':torch.cuda.synchronize()
                row=dict(step=state['step'],attempt=state['attempts'],updated=good,loss=total_loss,
                    grad_norm=norm,lr=lr,block=block,target_units=count,target_bytes=target_bytes,
                    seen_units=state['seen_units'],seen_bytes=state['seen_bytes'],fraction=state['seen_bytes']/budget,
                    skipped_attempts=state['skipped_attempts'],data_wait_seconds=ready-tick,
                    train_seconds=time.perf_counter()-step_start,seconds=time.perf_counter()-tick,
                    physical_positions=physical,mean_bytes_per_second=(state['seen_bytes']-start_bytes)/max(1e-9,time.perf_counter()-started))
                if provider.packer:row['packing_cache']=dict(provider.packer.stats)
                log.write(json.dumps(row)+'\n')
                if state['step']%tc['log_every']==0 or not good:
                    if observer:row['gpu']=observer.summary()
                    print(json.dumps(row),flush=True); write_json(run/'status.json',dict(event='training',**row))
                if invalid>=20:raise FloatingPointError('20 consecutive skipped updates; current targets retained for recovery')
                if good or stopped['value']:break
            if good:
                milestones=[m for m in tc['milestones'] if state['seen_bytes']>=budget*m and m not in state['saved_milestones']]
                for milestone in milestones:
                    state['saved_milestones'].append(milestone)
                    stage=run/f'stage_{milestone:g}.pt'
                    writer.submit(stage,payload(),guard,estimate).result()
                    if tc.get('stage_eval',False):
                        command=[sys.executable,str(code/'scripts/review_multilingual_prior.py'),
                                 '--checkpoint',str(stage),'--out',str(run/f'eval_{milestone:g}'),
                                 '--phase','teacher','--prefixes-per-language','1']
                        with (run/f'eval_{milestone:g}.log').open('a',encoding='utf8') as eval_log:
                            completed=subprocess.run(command,stdout=eval_log,stderr=subprocess.STDOUT)
                        print(json.dumps(dict(event='stage_evaluation',fraction=milestone,
                                              exit_code=completed.returncode)),flush=True)
                if state['step']%tc['save_every']==0 or state['step']==1:save()
            if stopped['value'] or (stop_after and state['step']-start_step>=stop_after):break
        save()
    except BaseException as exc:
        failure=f'{type(exc).__name__}: {exc}'
        try:
            if safe_model:save(force=True)
            else:print(json.dumps(dict(event='preserved_previous_checkpoint',reason='optimizer/EMA update interrupted')),flush=True)
        except Exception as saving:print(json.dumps(dict(event='save_error',error=str(saving))),flush=True)
        raise
    finally:
        stream.close(); log.close()
        try:writer.close()
        except Exception as exc:
            failure=failure or f'Checkpoint write failed: {exc}'
        gpu=observer.summary() if observer else None
        if observer:observer.close()
        for sig,handler in old.items():signal.signal(sig,handler)
        result=dict(event='failed' if failure else ('complete' if state['seen_bytes']==budget else 'paused'),
            error=failure,step=state['step'],seen_bytes=state['seen_bytes'],seen_units=state['seen_units'],
            budget_bytes=budget,elapsed_seconds=time.perf_counter()-started,skipped_attempts=state['skipped_attempts'],
            bytes_by_language=state['bytes_by_language'],units_by_block=state['units_by_block'],gpu=gpu)
        write_json(run/'status.json',result); print(json.dumps(result),flush=True)
    if failure:raise RuntimeError(failure)
    return result
