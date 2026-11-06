"""Apply the shared measured capacity to either already approved training arm."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from char_diff.utils import ROOT,load_config,write_json

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('config');p.add_argument('--capacity',required=True,type=Path);p.add_argument('--out',required=True,type=Path)
    p.add_argument('--joint-peak',action='store_true')
    a=p.parse_args();cfg=load_config(a.config)
    capacity=json.loads(a.capacity.read_text(encoding='utf8'));selected=capacity['selected']
    if selected['status']!='ok' or selected['parameters']!=77587504:
        raise ValueError('Capacity result differs from the approved 77.6M model')
    if a.out.exists():
        existing=json.loads(a.out.read_text(encoding='utf8'))
        baseline=load_config(a.config)
        for section in ('data','model','train','storage'):
            original=dict(baseline[section]);actual=dict(existing[section])
            for key in ('micro_batch','grad_checkpoint'):
                original.pop(key,None);actual.pop(key,None)
            if actual!=original:raise ValueError('Existing deployment differs: '+section)
        print(json.dumps(dict(event='reuse_frozen_deployment',config=str(a.out))),flush=True)
        raise SystemExit(0)
    if a.joint_peak:
        # Keep the AE busy through the first full update to measure the actual
        # simultaneous peak, on the device assigned to this training arm.
        micro=selected['micro_batch']
        while micro>=1:
            path=a.out.parent/f'joint-peak-{micro}.json'
            command=[sys.executable,str(Path(__file__).with_name('probe_multilingual_prior.py')),
                     a.config,'--worker',str(micro),'--out',str(path)]
            if selected['grad_checkpoint']:command.append('--checkpoint')
            subprocess.run(command,check=True,timeout=900)
            result=json.loads(path.read_text(encoding='utf8'))
            if result['status']=='error':raise RuntimeError(result['error'])
            if result['status']=='ok' and result['gpu']['peak_reserved_percent']<=94 and result['gpu'].get('nvidia_memory_peak_percent',0)<=96:
                selected=result;break
            micro-=2 if micro>4 else 1
        else:raise RuntimeError('No joint AE/prior working point fits the assigned GPU')
    cfg['train']['micro_batch']=selected['micro_batch'];cfg['model']['grad_checkpoint']=selected['grad_checkpoint']
    a.out.parent.mkdir(parents=True,exist_ok=True)
    write_json(a.out,cfg)
    print(json.dumps(dict(mode=cfg['data']['mode'],micro_batch=selected['micro_batch'],
        batch_windows=cfg['train']['batch_windows'],grad_checkpoint=selected['grad_checkpoint'],
        capacity_units_per_second=selected['units_per_second'])),flush=True)
