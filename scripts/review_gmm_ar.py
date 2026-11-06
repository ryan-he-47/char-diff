"""Shared dev next-unit EM and fixed free continuations for the fresh GMM AR arm."""
import argparse
import json
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from char_diff.ar import ARCache, GMMAR, history_batch, generate as ar_generate
from char_diff.codec import Codec
from char_diff.data import LatentCache
from char_diff.model import Denoiser
from char_diff.selection import CycleScorer, needs_retry, prefer_retry
from char_diff.utils import ROOT, sha256, write_json
from compare_byte_em import diffusion_batch, diffusion_predict
from review_block_size import metrics

ASSETS = ROOT/'assets/gmm_ar_eval_20261103'
OUT = ROOT/'runs/gmm_ar_review_20261103'
DOCS = ROOT/'docs/experiments/gmm-autoregressive'


def read(p): return json.loads(Path(p).read_text(encoding='utf-8'))
def lines(p): return [json.loads(s) for s in Path(p).read_text(encoding='utf-8').splitlines()]


def prepare():
    """Freeze the existing controls/prompts as tiny portable evaluation assets."""
    ASSETS.mkdir(parents=True,exist_ok=True)
    write_json(ASSETS/'anchors.json',read(ROOT/'runs/cross_model_byte_em_20261102/anchors.json'))
    write_json(ASSETS/'prompts.json',{'prompts':read(ROOT/'runs/review_715121/panel/prompts.json')['prompts'][:4]})
    controls = []
    for arm, file in [('x0_b1_2.25',ROOT/'runs/block_size_review_20261102/b1-teacher.jsonl'),
                      ('x0_b4_2.25',ROOT/'runs/block_size_review_20261102/b4-teacher.jsonl')]:
        controls.extend(dict(r,arm=arm) for r in lines(file))
    for r in lines(ROOT/'runs/cycle_retry_20261102/teacher.jsonl'):
        # Saved cycle rows carry both original and selected outcomes.
        controls.append(dict(r,arm='x0_b1_retry_2.25'))
    with (ASSETS/'controls.jsonl').open('w',encoding='utf-8') as f:
        for r in controls: f.write(json.dumps(r)+'\n')
    panel_ids={q['doc'] for q in read(ASSETS/'prompts.json')['prompts']}
    sample_controls=[]
    for r in lines(ROOT/'runs/sampling_steps_715121/samples.jsonl'):
        if r['doc'] in panel_ids and r['seed']==42 and r['steps']==32:
            sample_controls.append(dict(r,arm='x0_b16_2.0'))
    for r in lines(ROOT/'runs/block_size_review_20261102/samples.jsonl'):
        if r['arm']=='b1':sample_controls.append(dict(r,arm='x0_b1_2.25'))
    for r in lines(ROOT/'runs/cycle_retry_20261102/samples.jsonl'):
        sample_controls.append(dict(r,arm='x0_b1_retry_2.25'))
    with (ASSETS/'sample-controls.jsonl').open('w',encoding='utf-8') as f:
        for r in sample_controls:f.write(json.dumps(r,ensure_ascii=False)+'\n')
    provenance={'cache_fingerprint':LatentCache(ROOT/'data/tinystories').fingerprint,
               'anchors_sha256':sha256(ASSETS/'anchors.json'),
               'controls_sha256':sha256(ASSETS/'controls.jsonl'),
               'sample_controls_sha256':sha256(ASSETS/'sample-controls.jsonl'),
               'prompts_sha256':sha256(ASSETS/'prompts.json')}
    if (ASSETS/'x0_2epoch_teacher.jsonl').exists():
        provenance['x0_2epoch_teacher_sha256']=sha256(ASSETS/'x0_2epoch_teacher.jsonl')
        provenance['x0_2epoch_summary']=read(ASSETS/'x0_2epoch_teacher.summary.json')
    write_json(ASSETS/'provenance.json',provenance)


def score(case, seed, raw, **extra):
    target = bytes.fromhex(case['target_hex'])
    return {'id':case['id'],'window':case['window'],'seed':seed,'predicted_unit_hex':raw.hex(),
            'target_hex':target.hex(),'target_bytes':len(target),'exact':int(raw==target),**extra}


@torch.inference_mode()
def diffusion_reference(batch_size, limit=0):
    cache = LatentCache(ROOT/'data/tinystories','dev'); codec = Codec(device='cuda')
    checkpoint = ROOT/'runs/tinystories_x0_22m/last.pt'
    state = torch.load(checkpoint,map_location='cpu',weights_only=False)
    assert state['cache_fingerprint'] == cache.fingerprint and state['codec'] == codec.fingerprint
    model = Denoiser(state['config']['model']).cuda().eval(); model.load_state_dict(state['ema'])
    cases = read(ASSETS/'anchors.json')['cases']; cases = cases[:limit] if limit else cases
    streams = {s:torch.Generator(device='cuda').manual_seed(s) for s in (42,123)}
    mean = torch.tensor(cache.mean,device='cuda'); rows=[]; tick=time.perf_counter()
    dest = ASSETS/('x0_2epoch_smoke.jsonl' if limit else 'x0_2epoch_teacher.jsonl')
    with dest.open('x',encoding='utf-8',buffering=1) as f:
        for start in range(0,len(cases),batch_size):
            group=cases[start:start+batch_size]; packed=diffusion_batch(cache,group)
            for seed in (42,123):
                noise=torch.stack([torch.randn(16,48,device='cuda',generator=streams[seed]) for _ in group])
                z=diffusion_predict(model,packed,noise,'euler')
                raw=codec.decode(z[:,0]*cache.scale+mean)
                for c,v in zip(group,raw):
                    row=score(c,seed,v,arm='x0_b16_2.0');rows.append(row);f.write(json.dumps(row)+'\n')
            if start%(batch_size*8)==0: print(json.dumps({'reference_done':start+len(group),'total':len(cases)}),flush=True)
    write_json(dest.with_suffix('.summary.json'),{'checkpoint_sha256':sha256(checkpoint),'state':state['state'],
               'metrics':metrics(rows),'seconds':time.perf_counter()-tick,'weights':'ema','steps':32,'precision':'fp32'})


@torch.inference_mode()
def teacher(model,state,cache,codec,batch_size,limit=0):
    cases=read(ASSETS/'anchors.json')['cases']; cases=cases[:limit] if limit else cases
    streams={s:torch.Generator(device='cuda').manual_seed(s) for s in (42,123)}
    retries={s:torch.Generator(device='cuda').manual_seed(s+1000003) for s in (42,123)}
    scorer=CycleScorer(codec,state['normalizer']); mean=torch.tensor(cache.mean,device='cuda')
    rows=[]; tick=time.perf_counter(); nll_total=0.; nll_count=0
    with (OUT/'teacher.jsonl').open('x',encoding='utf-8',buffering=1) as f:
        for start in range(0,len(cases),batch_size):
            group=cases[start:start+batch_size]
            packed,lengths=history_batch([cache.vectors[c['history_ids']] for c in group],'cuda')
            hidden=model(**packed)[torch.arange(len(group),device='cuda'),torch.tensor(lengths,device='cuda')-1]
            # One mode enumeration per context; stochastic draws share this fixed density.
            peaks=model.head.modes(hidden)
            if len(group)==1: peaks=[peaks]
            for values in peaks:
                if not torch.isfinite(values[0]).all() or not torch.isfinite(values[2]).all():
                    raise FloatingPointError('Nonfinite density peaks')
            deterministic=model.head.mode(hidden)
            for c,raw in zip(group,codec.decode(deterministic)):
                row=score(c,0,raw,arm='gmm_mode'); rows.append(row);f.write(json.dumps(row)+'\n')
            for seed in (42,123):
                chosen=[]
                for modes,_,logdens in peaks:
                    idx=torch.multinomial(logdens.softmax(-1),1,generator=streams[seed])
                    chosen.append(modes[idx[0]])
                raw=torch.stack(chosen); z=(raw-mean)/cache.scale
                scores=scorer(z)
                for i,(c,first) in enumerate(zip(group,scores)):
                    original=bytes.fromhex(first['unit_hex'])
                    row=score(c,seed,original,arm='gmm_modes_peak');rows.append(row);f.write(json.dumps(row)+'\n')
                    retry=None; accepted=False; selected=first
                    if needs_retry(first,.1):
                        modes,_,logdens=peaks[i]
                        idx=torch.multinomial(logdens.softmax(-1),1,generator=retries[seed])
                        retry=scorer((modes[idx]-mean)/cache.scale)[0]
                        accepted=prefer_retry(first,retry)
                        if accepted:selected=retry
                    row=score(c,seed,bytes.fromhex(selected['unit_hex']),arm='gmm_modes_peak_retry',
                              triggered=retry is not None,accepted_retry=accepted,original=first,retry=retry)
                    rows.append(row);f.write(json.dumps(row)+'\n')
            # The true next latent enters only the diagnostic likelihood and final scoring.
            targets=[]
            for c in group:
                unit_id=cache.ids[int(cache.offsets[c['doc']])+c['anchor']]
                targets.append(cache.vectors[int(unit_id)])
            targets=torch.tensor(np.asarray(targets),device='cuda')*cache.scale+mean
            nll_total+=float(model.head.nll(hidden,targets).sum());nll_count+=len(group)
            print(json.dumps({'ar_teacher_done':start+len(group),'total':len(cases)}),flush=True)
    groups={name:[r for r in rows if r['arm']==name] for name in sorted({r['arm'] for r in rows})}
    if not limit:
        expected={(c['id'],s) for c in cases for s in (42,123)}
        for name in ('gmm_modes_peak','gmm_modes_peak_retry'):
            assert {(r['id'],r['seed']) for r in groups[name]}==expected
        for r in lines(ASSETS/'controls.jsonl'):
            groups.setdefault(r['arm'],[]).append(r)
        for r in lines(ASSETS/'x0_2epoch_teacher.jsonl'):
            groups.setdefault(r['arm'],[]).append(r)
    report={'models':{k:metrics(v) for k,v in groups.items()},'gmm_raw_ae_nll_per_unit':nll_total/nll_count,
            'seconds':time.perf_counter()-tick,'retry_count':sum(r.get('triggered',False) for r in rows),
            'accepted_retries':sum(r.get('accepted_retry',False) for r in rows)}
    if not limit:
        for name,values in groups.items():
            if name=='gmm_mode':continue
            assert len(values)==len(expected) and {(r['id'],r['seed']) for r in values}==expected
        windows=read(ASSETS/'anchors.json')['windows']; wi={w:i for i,w in enumerate(windows)}
        draws=np.random.default_rng(20261103).integers(0,len(windows),(5000,len(windows)))
        rates={}
        for name,values in groups.items():
            counts=np.zeros((len(windows),4))
            for r in values:counts[wi[r['window']]]+=[r['exact'],1,r['exact']*r['target_bytes'],r['target_bytes']]
            totals=counts[draws].sum(1);rates[name]=totals[:,[0,2]]/totals[:,[1,3]]
        report['paired_window_bootstrap']={'draws':5000,'seed':20261103,'delta_pp':{}}
        for arm,control in [('gmm_modes_peak','x0_b16_2.0'),('gmm_modes_peak','x0_b1_2.25'),
                            ('gmm_modes_peak_retry','x0_b1_retry_2.25')]:
            report['paired_window_bootstrap']['delta_pp'][arm+' vs '+control]={
                key:{'observed':100*(report['models'][arm][key]-report['models'][control][key]),
                     'ci95':np.quantile(100*(rates[arm][:,i]-rates[control][:,i]),[.025,.975]).tolist()}
                for i,key in enumerate(('unit_em','byte_weighted_unit_em'))}
    write_json(OUT/'teacher-summary.json',report)
    print(json.dumps({'teacher_summary':{k:{m:v[m] for m in ('unit_em','byte_weighted_unit_em')}
                                        for k,v in report['models'].items()}}),flush=True)


@torch.inference_mode()
def samples(model,state,cache,codec,limit=0):
    panel=read(ASSETS/'prompts.json')['prompts']; panel=panel[:1] if limit else panel
    offsets=np.fromfile(cache.root/'unit_offsets.u64',dtype='<u8')
    blob=(cache.root/'units.bin').read_bytes();counts=np.fromfile(cache.root/'train_counts.u64',dtype='<u8')
    seen={blob[int(offsets[i]):int(offsets[i+1])] for i in np.flatnonzero(counts)}
    vectors={};rows=[]
    with (OUT/'samples.jsonl').open('x',encoding='utf-8',buffering=1) as f:
        for history in ('continuous','reencode'):
            for q in panel:
                torch.cuda.synchronize();tick=time.perf_counter()
                with torch.autocast('cuda',dtype=torch.float16):
                    z,parts=ar_generate(model,cache.story(q['doc'])[:32],4 if limit else 128,
                                        state['normalizer'],codec,seed=42,history=history)
                torch.cuda.synchronize();elapsed=time.perf_counter()-tick;raw=b''.join(parts)
                assert torch.isfinite(z).all()
                row={'arm':'gmm_modes_peak','history':history,'doc':q['doc'],'seed':42,'prompt':q['prompt'],
                     'units':len(parts),'seconds':elapsed,'generated':raw.decode('utf-8',errors='replace'),
                     'generated_hex':raw.hex(),'unit_hex':[p.hex() for p in parts],
                     'seen_training_units':sum(p in seen for p in parts),'empty_units':sum(not p for p in parts),
                     'utf8_valid':raw.decode('utf-8',errors='replace').encode('utf-8')==raw}
                rows.append(row);f.write(json.dumps(row,ensure_ascii=False)+'\n')
                vectors[f'{history}_doc{q["doc"]}']=z.cpu().numpy()
                np.savez(OUT/'sample-latents.npz',**vectors)
                print(json.dumps({'sample_done':history,'doc':q['doc'],'seconds':elapsed}),flush=True)
    write_json(OUT/'sample-summary.json',{'groups':{h:{'cases':sum(r['history']==h for r in rows),
        'units':sum(r['units'] for r in rows if r['history']==h),
        'seen_training_units':sum(r['seen_training_units'] for r in rows if r['history']==h),
        'mean_seconds':float(np.mean([r['seconds'] for r in rows if r['history']==h]))}
        for h in ('continuous','reencode')}})


def summarize():
    """Portable machine totals and a reviewable, labelled side-by-side text report."""
    protocol=read(OUT/'protocol.json'); report=read(OUT/'teacher-summary.json')
    DOCS.mkdir(parents=True,exist_ok=True)
    body=['# GMM 自回归共同 dev 验收','',
          '固定64 dev窗口、1,845锚点；随机读出两种子共3,690次，确定性mode为1,845次。EMA，FP32教师强制。',
          '完整首unit EM严格比较整个字节串；字节加权按真实目标长度。两遍GMM与两遍B16对齐数据量，2.25遍模型作为当前系统参照。','',
          '| 模型 / 读出 | unit EM | 字节加权unit EM |','|---|---:|---:|']
    for name,value in report['models'].items():
        body.append(f'| {name} | {100*value["unit_em"]:.2f}% | {100*value["byte_weighted_unit_em"]:.2f}% |')
    body+=['',f'GMM完成{protocol["state"]["step"]:,}次成功更新，{protocol["state"]["seen_units"]:,}个有效目标。',
           'GMM modes_peak为密度峰高softmax、T=1无截断；mode单列确定性结果；retry为R>0.1、最多一次重试。',
           '训练原始AE NLL和各组配对窗口bootstrap区间保存在teacher-summary.json。',
           '自动评测已完成；正文质量及最终研究结论由随后人工评审回填。','']
    text='\n'.join(body);(OUT/'results.md').write_text(text,encoding='utf-8')
    (DOCS/'results.md').write_text(text,encoding='utf-8')
    pairs=['# 固定前缀续写对照','',
           '前缀32units、续写128units、seed42；FP16模型/FP32字节读出。GMM记录连续和重编码历史，扩散使用缓存结果。',
           '耗时记录在原始JSON；不同机器上的数值分别保留设备信息。','']
    controls=lines(ASSETS/'sample-controls.jsonl');rows=lines(OUT/'samples.jsonl')
    for q in read(ASSETS/'prompts.json')['prompts']:
        pairs += [f'## doc {q["doc"]}','','**前缀**','',q['prompt'],'']
        for r in [v for v in controls+rows if v['doc']==q['doc']]:
            pairs += [f'**{r["arm"]} / {r.get("history","continuous")}**','',r['generated'],'']
    text='\n'.join(pairs);(OUT/'samples.md').write_text(text,encoding='utf-8')
    (DOCS/'samples.md').write_text(text,encoding='utf-8')


def main():
    global OUT
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase',choices=['prepare','reference','teacher','samples','summarize','all'],default='all')
    p.add_argument('--checkpoint',default='runs/tinystories_gmm_ar_22m/last.pt')
    p.add_argument('--output',default=str(OUT));p.add_argument('--batch',type=int,default=32)
    p.add_argument('--limit',type=int,default=0,help='Small affected-path smoke; outputs use a separate directory')
    a=p.parse_args();torch.set_num_threads(4);OUT=Path(a.output);OUT.mkdir(parents=True,exist_ok=True)
    if a.phase=='prepare':prepare();return
    if a.phase=='reference':diffusion_reference(a.batch,a.limit);return
    if a.phase=='summarize':summarize();return
    state=torch.load(ROOT/a.checkpoint,map_location='cpu',weights_only=False)
    cache=ARCache(ROOT/'data/tinystories','dev');codec=Codec(device='cuda')
    assert state['cache_fingerprint']==cache.fingerprint and state['codec']==codec.fingerprint
    provenance=read(ASSETS/'provenance.json')
    assert provenance['cache_fingerprint']==cache.fingerprint
    if not a.limit:
        for key,name in [('anchors','anchors.json'),('controls','controls.jsonl'),
                         ('sample_controls','sample-controls.jsonl'),('prompts','prompts.json'),
                         ('x0_2epoch_teacher','x0_2epoch_teacher.jsonl')]:
            assert provenance[key+'_sha256']==sha256(ASSETS/name)
    if not a.limit and state['state']['seen_units']<state['budget_units']:
        raise ValueError('Final acceptance requires the completed two-epoch training budget')
    model=GMMAR(state['config']['model']).cuda().eval();model.load_state_dict(state['ema'])
    write_json(OUT/'protocol.json',{'checkpoint':a.checkpoint,'checkpoint_sha256':sha256(ROOT/a.checkpoint),
        'state':state['state'],'parameters':sum(p.numel() for p in model.parameters()),'weights':'ema',
        'readout':'legacy density modes, T=1, no truncation; MAP separately',
        'teacher':'same true histories, strict full-unit EM and byte-weighted full-unit EM, FP32, seed42/123',
        'retry':'same R>0.1, at most one independent retry, lower R valid candidate; additional column',
        'samples':'4 fixed prefixes, 128 units, seed42, FP16 backbone/FP32 readout+codec; continuous and legacy reencoded history',
        'controls':'data-matched 2epoch B16; current 2.25epoch B1/B4 and B1 retry labelled separately',
        'device':torch.cuda.get_device_name(),
        'cache_fingerprint':cache.fingerprint,'assets':read(ASSETS/'provenance.json'),'smoke_limit':a.limit})
    if a.phase in ('teacher','all'):teacher(model,state,cache,codec,a.batch,a.limit)
    if a.phase in ('samples','all'):samples(model,state,cache,codec,a.limit)
    if a.phase=='all' and not a.limit:summarize()


if __name__=='__main__':main()
