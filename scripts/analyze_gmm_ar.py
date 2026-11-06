"""Summarize cached GMM AR acceptance; no new model inference."""
import json
import re
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from char_diff.utils import ROOT, sha256, write_json

def read(p):return json.loads(p.read_text(encoding='utf-8'))
def lines(p):return [json.loads(s) for s in p.read_text(encoding='utf-8').splitlines()]

def main():
    out=ROOT/'runs/gmm_ar_review_20261103';run=ROOT/'runs/tinystories_gmm_ar_22m'
    protocol=read(out/'protocol.json')
    assert sha256(run/'last.pt')==protocol['checkpoint_sha256']
    raw=lines(run/'train.jsonl'); breaks=[i for i in range(1,len(raw)) if raw[i]['attempt']<=raw[i-1]['attempt']]
    assert len(breaks)==1
    split=breaks[0];first,second=raw[:split],raw[split:]
    restored_attempt=second[0]['attempt']-1
    canonical=[r for r in first if r['attempt']<=restored_attempt]+second
    assert len(canonical)==protocol['state']['attempts']
    assert sum(r['updated'] for r in canonical)==protocol['state']['step']
    updated=[r for r in canonical if r['updated']]
    diagnostic=[r for r in updated if 'pi_entropy' in r]
    tail=[r for r in diagnostic if r['step']>protocol['state']['step']-500]
    train={'state':protocol['state'],'checkpoint_sha256':protocol['checkpoint_sha256'],
           'resume_step':second[0]['step']-int(second[0]['updated']),
           'resume_attempt':restored_attempt,'replayed_log_rows':len(raw)-len(canonical),
           'skips':sum(not r['updated'] for r in canonical),
           'resume_skips':sum(not r['updated'] for r in second),
           'resume_successful_updates':sum(r['updated'] for r in second),
           'last500_loss':float(np.mean([r['loss'] for r in updated[-500:]])),
           'last200_loss':float(np.mean([r['loss'] for r in updated[-200:]])),
           'loss_by_successful_epoch':{},
           'last500_diagnostics':{k:float(np.mean([r[k] for r in tail])) for k in
                                  ('pi_entropy','sigma_min','sigma_mean','sigma_near_floor_fraction')}}
    for lo,hi in ((0,.25),(.25,.5),(.5,1),(1,1.5),(1.5,1.8),(1.8,2.01)):
        vals=[r['loss'] for r in updated if lo<=r['epochs']<hi]
        train['loss_by_successful_epoch'][f'{lo}-{hi}']={'updates':len(vals),'mean':float(np.mean(vals))}
    rows=lines(out/'teacher.jsonl');groups={}
    for r in rows:groups.setdefault(r['arm'],[]).append(r)
    mode={r['id']:r for r in groups['gmm_mode']}
    sampled={(r['id'],r['seed']):r for r in groups['gmm_modes_peak']}
    retry=groups['gmm_modes_peak_retry']
    a=sum(mode[r['id']]['exact'] and not r['exact'] for r in sampled.values())
    b=sum(not mode[r['id']]['exact'] and r['exact'] for r in sampled.values())
    cycle=[r['original']['cycle_rmse'] for r in retry]
    choices={'mode_right_sample_wrong':a,'mode_wrong_sample_right':b,'sample_net_hits_vs_mode_replicated':b-a,
             'retry_triggered':sum(r['triggered'] for r in retry),'retry_accepted':sum(r['accepted_retry'] for r in retry),
             'retry_bytes_changed':sum(r['predicted_unit_hex']!=sampled[r['id'],r['seed']]['predicted_unit_hex'] for r in retry),
             'retry_wrong_to_right':sum(r['exact'] and not sampled[r['id'],r['seed']]['exact'] for r in retry),
             'retry_right_to_wrong':sum(not r['exact'] and sampled[r['id'],r['seed']]['exact'] for r in retry),
             'valid_initial':sum(r['original']['valid'] for r in retry),
             'cycle_rmse_quantiles':dict(zip(('p50','p90','p99','max'),map(float,np.quantile(cycle,[.5,.9,.99,1]))))}
    samples=lines(out/'samples.jsonl')+lines(ROOT/'assets/gmm_ar_eval_20261103/sample-controls.jsonl')
    free={}
    for r in samples:
        key=r['arm']+'/'+r.get('history','continuous')
        v=free.setdefault(key,{'cases':0,'units':0,'seen_training_units':0,'empty_units':0,'utf8_valid':0,
                               'fourgrams':0,'repeated_fourgrams':0})
        for field in ('units','seen_training_units','empty_units','utf8_valid'):v[field]+=r[field]
        v['cases']+=1
        words=re.findall(r"\b\w+\b",r['generated'].lower())
        ngrams=[tuple(words[i:i+4]) for i in range(len(words)-3)]
        v['fourgrams']+=len(ngrams);v['repeated_fourgrams']+=len(ngrams)-len(set(ngrams))
    for v in free.values():v['repeated_fourgram_fraction']=v['repeated_fourgrams']/v['fourgrams']
    report={'training':train,'readout_choices':choices,'free_samples':free,
            'teacher':read(out/'teacher-summary.json'),
            'notes':['Canonical trajectory drops the interrupted replay tail; raw logs preserved.',
                     'Seen-string rate and repeated 4-grams describe this fixed four-prompt panel.',
                     'Generation timings were collected on different devices; no cross-device speedup claim.']}
    write_json(out/'final-analysis.json',report)
    write_json(ROOT/'docs/experiments/gmm-autoregressive/evidence/final-analysis.json',report)
    print(json.dumps({k:report[k] for k in ('training','readout_choices','free_samples')},ensure_ascii=False))

if __name__=='__main__':main()
