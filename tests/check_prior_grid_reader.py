"""Exercise new permanent-shard lookup/hot-cache wiring and a full-model update."""
from copy import deepcopy
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from char_diff.bpe_packing import FrozenUnits,BPEPacker
from char_diff.flow import corrupt,loss_sum
from char_diff.model import Denoiser
from char_diff.prior_stream import RawCorpus,GroupProvider,UnitStream,collate
from char_diff.segmented_cache import write_segmented_shard
from char_diff.utils import ROOT,load_config,write_json


def main():
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
    out=ROOT/'runs/multilingual_prior_build_20261106'
    cfg=load_config(ROOT/'configs/multilingual_x0_78m_grid8.yaml')
    cfg['data'].update(manifest=str(out/'raw/manifest.json'),
        ae_checkpoint='runs/full_ae_review_20261106/last.pt',segments=str(out/'grid_fixture'),
        hot_cache=str(out/'grid_hot'),hot_cache_gib=.002,window=32,carry=8,lanes=3)
    encoder=FrozenUnits(ROOT/cfg['data']['ae_checkpoint'],'cuda',1024)
    packer=BPEPacker(out/'bpe_assets',encoder,128)
    raw=RawCorpus(cfg['data']['manifest']);records=[];expected={}
    for gi in range(len(raw.groups)):
        texts,coords,lang=raw.read(gi)
        for text,coordinate,(units,z) in zip(texts,coords,packer.pack(texts)):
            records.append(dict(raw=text.encode(),lengths=list(map(len,units)),language=lang,
                                source_file=coordinate[0],source_row=coordinate[1]))
            expected[coordinate]=units
    root=Path(cfg['data']['segments']);root.mkdir(exist_ok=True)
    provenance=dict(languages=sorted({r['language'] for r in records}),ae_sha256=cfg['data']['ae_sha256'],
                    purpose='reader-only fixture using already verified boundaries')
    if not (root/'segments/shard-000000').exists():
        write_segmented_shard(root/'segments/shard-000000',records,provenance,
            metadata=dict(before_cursor=[0,0],after_state=dict(cursor=[len(raw.spec['parts']),0],shards=1)))
    write_json(root/'segmentation.json',provenance)
    write_json(root/'progress.json',dict(state=dict(shards=1,complete=True)))
    provider=GroupProvider(cfg,encoder,ROOT)
    for gi in range(len(raw.groups)):
        a=provider.load(gi);b=provider.load(gi)
        for row,cached in zip(a,b):
            assert row[0]==expected[row[3]] and np.array_equal(row[1],cached[1])
    # Hot cache reuse after process-style reopening.
    second=GroupProvider(cfg,encoder,ROOT)
    assert np.array_equal(provider.load(0)[0][1],second.load(0)[0][1])
    provider.close();second.close();raw.close();del packer,encoder
    torch.cuda.empty_cache()
    net=Denoiser(cfg['model']).cuda().train()
    opt=torch.optim.AdamW(net.parameters(),lr=3e-4)
    scaler=torch.amp.GradScaler('cuda',init_scale=128)
    rng=np.random.default_rng(31)
    part=dict(z=rng.normal(size=(640,48)).astype('f2'),prior=0,start=128,end=640,phase=3)
    batch=collate([[part]],8,dict(mean=[0.]*48,scale=1.),'cuda')
    x,t=corrupt(batch)
    with torch.autocast('cuda',dtype=torch.float16):
        prediction=net(x,t,batch['roles'],batch['positions'],batch['lengths'],batch['allowed'])
        raw_loss,n=loss_sum(prediction,batch);loss=raw_loss/(48*n)
    scaler.scale(loss).backward();scaler.unscale_(opt)
    norm=torch.nn.utils.clip_grad_norm_(net.parameters(),1.)
    assert torch.isfinite(norm) and torch.isfinite(loss)
    scaler.step(opt);scaler.update()
    result=dict(documents=len(records),raw_bytes=sum(len(r['raw']) for r in records),
        permanent_coordinates_and_hot_cache=True,parameters=sum(p.numel() for p in net.parameters()),
        precision='fp16',full_window_targets=n,full_model_loss=float(loss),grad_norm=float(norm),
        allocated_bytes=torch.cuda.max_memory_allocated())
    write_json(out/'grid_and_78m_smoke.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':main()
