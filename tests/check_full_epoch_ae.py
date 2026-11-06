"""Exercise full-pass termination, batch retry cursor, resume and LR endpoints."""
import json
import sys
import tempfile
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from char_diff.ae_training import train
from char_diff.ae_prefetch import epoch_lr
from char_diff.utils import ROOT,load_config,sha256,write_json


def main():
    cfg=load_config(ROOT/'configs/full_ae48_epoch1.yaml')
    parent=ROOT/'runs/utf8_ae_wsd_20261103/training_smoke/stable_continuous/last.pt'
    saved=torch.load(parent,map_location='cpu',weights_only=False)
    cfg['model']=saved['cfg']['model']
    with tempfile.TemporaryDirectory(dir=ROOT/'runs') as temp:
        root=Path(temp)
        docs=['a\0b é果🙂'*7,'xy浊嗒童'*5,'end']
        part=root/'raw.parquet'
        pq.write_table(pa.table({'story':docs}),part,row_group_size=1)
        total=sum(len(t.encode()) for t in docs)
        write_json(root/'manifest.json',dict(status='complete',format='full-raw-ae-v1',raw_bytes=total,
                    parts=[dict(file=part.name,language='fixture',groups=[1]*len(docs),sha256=sha256(part),
                                characters=''.join(sorted(set(''.join(docs))))) ]))
        cfg['storage'].update(require_quota=False,project_limit_gib=1000)
        cfg['data'].update(init_stable=str(parent),init_sha256=sha256(parent),full_manifest=str(root/'manifest.json'))
        cfg['train'].update(batch_windows=3,micro_batch=3,precision='fp32',save_every=25,run_dir=str(root/'a'))
        train(cfg,device='cpu')
        cfg['train']['run_dir']=str(root/'b')
        train(cfg,device='cpu',stop_after=1)
        out=train(cfg,device='cpu',resume=root/'b/last.pt')
        a,b=[torch.load(root/sub/'last.pt',map_location='cpu',weights_only=False) for sub in ('a','b')]
        assert a['sampler']==b['sampler'] and a['state']==b['state']
        assert all(torch.equal(v,b['model'][k]) for k,v in a['model'].items())
        assert out['event']=='complete' and out['sampling']['natural_source_bytes']==total
        assert a['optimizer']['param_groups'][0]['lr']==cfg['train']['final_lr']
        assert epoch_lr(90,100,1.,0.)==1. and epoch_lr(95,100,1.,0.)==.5
        print('PASS exact one pass, final partial batch, resume, stable endpoint, decay to floor',total)
if __name__=='__main__':main()
