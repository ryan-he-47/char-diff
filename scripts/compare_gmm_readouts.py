"""Approved fixed-weight mode/T=.6 diagnosis, reusing the existing T=1 controls."""
import argparse
from collections import Counter
import json
import re
import sys
import time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from char_diff.ar import GMMAR, ARCache, history_batch, generate
from char_diff.codec import Codec
from char_diff.utils import ROOT,sha256,write_json
from review_block_size import metrics

OLD=ROOT/'runs/gmm_ar_review_20261103'
OUT=ROOT/'runs/gmm_ar_readout_20261103'
ASSETS=ROOT/'assets/gmm_ar_eval_20261103'
DOCS=ROOT/'docs/experiments/gmm-autoregressive'
CHECKPOINT=ROOT/'runs/tinystories_gmm_ar_22m/last.pt'

def read(path):return json.loads(path.read_text(encoding='utf-8'))
def lines(path):return [json.loads(v) for v in path.read_text(encoding='utf-8').splitlines()]


@torch.inference_mode()
def teacher(model,cache,codec,batch_size=32):
    cases=read(ASSETS/'anchors.json')['cases']
    old={(r['id'],r['seed']):r for r in lines(OLD/'teacher.jsonl') if r['arm']=='gmm_modes_peak'}
    streams={s:torch.Generator(device='cuda').manual_seed(s) for s in (42,123)}
    controls={s:torch.Generator(device='cuda').manual_seed(s) for s in (42,123)}
    rows=[];differences=[];tick=time.perf_counter()
    with (OUT/'teacher-t06.jsonl').open('x',encoding='utf-8',buffering=1) as f:
        for start in range(0,len(cases),batch_size):
            group=cases[start:start+batch_size]
            packed,lengths=history_batch([cache.vectors[c['history_ids']] for c in group],'cuda')
            h=model(**packed)[torch.arange(len(group),device='cuda'),torch.tensor(lengths,device='cuda')-1]
            peaks=model.head.modes(h)
            if len(group)==1:peaks=[peaks]
            for seed in (42,123):
                selected=[];control=[]
                for centers,_,ld in peaks:
                    if not torch.isfinite(centers).all() or not torch.isfinite(ld).all():
                        raise FloatingPointError('Nonfinite density peaks')
                    selected.append(centers[torch.multinomial((ld/.6).softmax(-1),1,generator=streams[seed])[0]])
                    control.append(centers[torch.multinomial(ld.softmax(-1),1,generator=controls[seed])[0]])
                raw=codec.decode(torch.stack(selected));base=codec.decode(torch.stack(control))
                for c,v,b in zip(group,raw,base):
                    if b.hex()!=old[c['id'],seed]['predicted_unit_hex']:
                        differences.append({'id':c['id'],'seed':seed,'old':old[c['id'],seed]['predicted_unit_hex'],'new':b.hex()})
                    target=bytes.fromhex(c['target_hex'])
                    row={'id':c['id'],'window':c['window'],'seed':seed,'target_hex':target.hex(),
                         'target_bytes':len(target),'predicted_unit_hex':v.hex(),'exact':int(v==target)}
                    rows.append(row);f.write(json.dumps(row)+'\n')
            if start%(batch_size*16)==0:print(json.dumps({'teacher_done':start+len(group)}),flush=True)
    old_summary=read(OLD/'teacher-summary.json')['models']
    summary={'models':{'t1':old_summary['gmm_modes_peak'],'t06':metrics(rows),'mode':old_summary['gmm_mode']},
             't1_cached_byte_reproduction':{'cases':len(old),'differences':differences},'seconds':time.perf_counter()-tick}
    # Pair both seeds and adjacent targets within the same sampled dev window.
    windows=read(ASSETS/'anchors.json')['windows'];index={w:i for i,w in enumerate(windows)}
    counts=np.zeros((len(windows),4))
    for r in rows:
        delta=r['exact']-old[r['id'],r['seed']]['exact']
        counts[index[r['window']]]+=[delta,1,delta*r['target_bytes'],r['target_bytes']]
    totals=counts[np.random.default_rng(20261103).integers(0,len(windows),(5000,len(windows)))].sum(1)
    summary['t06_minus_t1_pp']={key:{'observed':100*(summary['models']['t06'][key]-summary['models']['t1'][key]),
         'ci95':np.quantile(100*totals[:,i]/totals[:,i+1],[.025,.975]).tolist()}
         for i,key in ((0,'unit_em'),(2,'byte_weighted_unit_em'))}
    write_json(OUT/'teacher-summary.json',summary)
    print(json.dumps({'teacher_summary':{k:{m:v[m] for m in ('unit_em','byte_weighted_unit_em')}
                                        for k,v in summary['models'].items()},'t1_differences':len(differences)}),flush=True)


@torch.inference_mode()
def samples(model,state,cache,codec):
    panel=read(ASSETS/'prompts.json')['prompts']
    old={r['doc']:r for r in lines(OLD/'samples.jsonl') if r['history']=='continuous'}
    # The sole generation code change forwards temperature; verify the default on a short real prefix.
    with torch.autocast('cuda',dtype=torch.float16):
        _,part=generate(model,cache.story(panel[0]['doc'])[:32],8,state['normalizer'],codec,seed=42)
    expected=old[panel[0]['doc']]['unit_hex'][:8]
    check={'default_doc':panel[0]['doc'],'units':8,'byte_exact':list(map(bytes.hex,part))==expected,
           'old_device':read(OLD/'protocol.json')['device'],'current_device':torch.cuda.get_device_name()}
    write_json(OUT/'targeted-check.json',check)
    if not check['byte_exact']:raise RuntimeError('Default short control differs; inspect device sensitivity before proceeding')
    offsets=np.fromfile(cache.root/'unit_offsets.u64',dtype='<u8');blob=(cache.root/'units.bin').read_bytes()
    counts=np.fromfile(cache.root/'train_counts.u64',dtype='<u8')
    seen={blob[int(offsets[i]):int(offsets[i+1])] for i in np.flatnonzero(counts)}
    vectors={}
    with (OUT/'samples.jsonl').open('x',encoding='utf-8',buffering=1) as f:
        for arm,kind,temperature in [('mode','mode',1.0),('t06','modes_peak',.6)]:
            for q in panel:
                torch.cuda.synchronize();tick=time.perf_counter()
                with torch.autocast('cuda',dtype=torch.float16):
                    z,parts=generate(model,cache.story(q['doc'])[:32],128,state['normalizer'],codec,
                                     seed=42,history='continuous',kind=kind,temperature=temperature)
                torch.cuda.synchronize();elapsed=time.perf_counter()-tick;raw=b''.join(parts)
                assert torch.isfinite(z).all()
                row={'arm':arm,'kind':kind,'temperature':temperature if kind=='modes_peak' else None,
                     'doc':q['doc'],'prompt':q['prompt'],'seed':42,'units':len(parts),'history':'continuous',
                     'generated':raw.decode('utf-8',errors='replace'),'generated_hex':raw.hex(),
                     'unit_hex':[p.hex() for p in parts],'seen_training_units':sum(p in seen for p in parts),
                     'empty_units':sum(not p for p in parts),'utf8_valid':raw.decode('utf-8',errors='replace').encode('utf-8')==raw,
                     'seconds':elapsed}
                f.write(json.dumps(row,ensure_ascii=False)+'\n')
                vectors[f'{arm}_doc{q["doc"]}']=z.cpu().numpy();np.savez(OUT/'sample-latents.npz',**vectors)
                print(json.dumps({'sample_done':arm,'doc':q['doc'],'seconds':elapsed,'text':row['generated']},ensure_ascii=True),flush=True)


def summarize():
    rows=[dict(r,arm='t1') for r in lines(OLD/'samples.jsonl') if r['history']=='continuous']+lines(OUT/'samples.jsonl')
    report={'groups':{},'cases':[]}
    for r in rows:
        words=re.findall(r'\b\w+\b',r['generated'].lower())
        grams=Counter(tuple(words[i:i+4]) for i in range(len(words)-3));total=sum(grams.values())
        row={'arm':r['arm'],'doc':r['doc'],'words':len(words),'fourgrams':total,
             'repeated_fourgrams':total-len(grams),'repeat_fraction':(total-len(grams))/total,
             'most_repeated_fourgrams':[{'text':' '.join(k),'count':v} for k,v in grams.most_common(3)]}
        report['cases'].append(row)
        g=report['groups'].setdefault(r['arm'],dict(cases=0,units=0,seen_training_units=0,empty_units=0,
                                                   utf8_valid=0,fourgrams=0,repeated_fourgrams=0))
        g['cases']+=1
        for key in ('units','seen_training_units','empty_units','utf8_valid'):g[key]+=r[key]
        for key in ('fourgrams','repeated_fourgrams'):g[key]+=row[key]
    for v in report['groups'].values():v['repeat_fraction']=v['repeated_fourgrams']/v['fourgrams']
    write_json(OUT/'sample-summary.json',report)
    panel=read(ASSETS/'prompts.json')['prompts']
    text=['# 固定GMM权重：T=1、T=0.6与确定性mode','',
          '同4前缀、前缀32units、续写128units、EMA，连续latent历史，FP16骨干/FP32头与AE。随机臂seed42，无截断；T1来自既有V100S缓存，新增两臂在本机运行。','']
    for q in panel:
        text += [f'## doc {q["doc"]}','','**前缀**','',q['prompt'],'']
        for arm in ('t1','t06','mode'):
            r=next(r for r in rows if r['doc']==q['doc'] and r['arm']==arm)
            text += [f'**{arm}**','',r['generated'],'']
    content='\n'.join(text)
    (OUT/'samples.md').write_text(content,encoding='utf-8')
    (DOCS/'readout-samples.md').write_text(content,encoding='utf-8')
    print(json.dumps({'sample_summary':report['groups']}),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase',choices=['all','teacher','samples','summarize'],default='all')
    a=p.parse_args();torch.set_num_threads(4);OUT.mkdir(parents=True,exist_ok=True)
    if a.phase=='summarize':summarize();return
    digest=sha256(CHECKPOINT);assert digest==read(OLD/'protocol.json')['checkpoint_sha256']
    state=torch.load(CHECKPOINT,map_location='cpu',weights_only=False)
    cache=ARCache(ROOT/'data/tinystories','dev');codec=Codec(device='cuda')
    assert state['cache_fingerprint']==cache.fingerprint and state['codec']==codec.fingerprint
    model=GMMAR(state['config']['model']).cuda().eval();model.load_state_dict(state['ema'])
    write_json(OUT/'protocol.json',{'checkpoint_sha256':digest,'weights':'ema','device':torch.cuda.get_device_name(),
        'source_sha256':{f:sha256(ROOT/f) for f in ('char_diff/ar.py','scripts/compare_gmm_readouts.py')},
        'teacher':'FP32, same 1845 dev anchors, seed42/123; new T=.6, cached mode/T1',
        'samples':'FP16 backbone, FP32 head/AE, 4 prefixes x mode/T=.6, 128 units, continuous history, seed42',
        'controls':'Existing T1 teacher/free outputs reused; short default generation and cost-free same-peak T1 teacher comparison',
        'mode':'Existing 24-step single-start approximate mode, kept distinct from low-T multi-start peak sampling',
        'anchors_sha256':sha256(ASSETS/'anchors.json'),'prompts_sha256':sha256(ASSETS/'prompts.json')})
    if a.phase in ('all','teacher'):teacher(model,cache,codec)
    if a.phase in ('all','samples'):samples(model,state,cache,codec);summarize()

if __name__=='__main__':main()
