"""Targeted coverage: packing, UTF-8/byte ledger, mixed masks/KV, exact resume."""
from collections import defaultdict
from copy import deepcopy
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from char_diff.bpe_packing import FrozenUnits,prepare_assets,BPEPacker
from char_diff.flow import generate
from char_diff.model import Denoiser,NOISY,CLEAN,BOS
from char_diff.prior_stream import GroupProvider,UnitStream,collate
from char_diff.prior_training import train,byte_lr
from char_diff.utils import ROOT,load_config,write_json,sha256

OUT=ROOT/'runs/multilingual_prior_build_20261106'


def same(a,b):
    if isinstance(a,torch.Tensor):return torch.equal(a.cpu(),b.cpu())
    if isinstance(a,np.ndarray):return np.array_equal(a,b)
    if isinstance(a,dict):return a.keys()==b.keys() and all(same(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)):return len(a)==len(b) and all(same(x,y) for x,y in zip(a,b))
    return a==b


def main():
    torch.set_num_threads(2); torch.backends.cuda.matmul.allow_tf32=False
    OUT.mkdir(parents=True,exist_ok=True)
    documents=json.loads((ROOT/'runs/bpe_pack_20261106/panels.json').read_text(encoding='utf8'))['test']
    documents=[row for i,row in enumerate(documents) if i%4<2]
    checkpoint=ROOT/'runs/full_ae_review_20261106/last.pt'
    encoder=FrozenUnits(checkpoint,'cuda',1024)
    assets=OUT/'bpe_assets'
    prepare_assets(ROOT/'assets/tokenizer.json',checkpoint,ROOT/'runs/bpe_pack_20261106/vocab_scores.npz',assets,encoder)
    packer=BPEPacker(assets,encoder,capacity=128)
    texts=[d['text'] for d in documents]
    cold=packer.pack(texts); warm=packer.pack(texts)
    uncached=BPEPacker(assets,encoder,capacity=0).pack(texts)
    for text,a,b,c in zip(texts,cold,warm,uncached):
        assert b''.join(a[0])==text.encode() and a[0]==b[0]==c[0]
        assert np.array_equal(a[1],b[1]) and np.array_equal(a[1],c[1])
        for unit in a[0]:unit.decode('utf8')
    # Compare only the changed production path to the already measured prototype.
    sys.path.insert(0,str(ROOT/'runs/bpe_pack_20261106'))
    from analyze import Probe,pruned_tokenizer,byte_inverse,pack_documents,RECIPES
    reference=Probe(); spec=json.loads((ROOT/'assets/tokenizer.json').read_text(encoding='utf8'))
    inverse=byte_inverse(); vocab={w:bytes(inverse[c] for c in w) for w in spec['model']['vocab']}
    valid=[u for u in vocab.values() if 0<len(u)<=32 and __import__('analyze').valid_utf8(u)]
    reference.cache.update(zip(valid,np.load(ROOT/'runs/bpe_pack_20261106/vocab_scores.npz')['metrics']))
    tokenizer,_=pruned_tokenizer(spec,vocab,reference,RECIPES[2])
    expected,_=pack_documents(documents,tokenizer,inverse,reference,RECIPES[2])
    assert [x[0] for x in cold]==expected
    del reference,packer
    raw=OUT/'raw'; raw.mkdir(exist_ok=True)
    langs=defaultdict(list)
    for doc in documents:langs[doc['language']].append(doc['text'])
    parts=[]
    for i,(lang,stories) in enumerate(sorted(langs.items())):
        path=raw/f'raw-{i:03d}.parquet'; pq.write_table(pa.table({'story':stories}),path,row_group_size=2)
        parts.append(dict(file=path.name,sha256=sha256(path),language=lang,groups=[len(stories)],
                          counts=dict(raw_bytes=sum(len(s.encode()) for s in stories),accepted=len(stories))))
    total=sum(p['counts']['raw_bytes'] for p in parts)
    write_json(raw/'manifest.json',dict(format='full-raw-ae-v1',status='complete',parts=parts,raw_bytes=total))
    cfg=load_config(ROOT/'configs/multilingual_x0_78m_bpe.yaml')
    cfg['data'].update(manifest=str(raw/'manifest.json'),ae_checkpoint=str(checkpoint),
        vocab_scores='runs/bpe_pack_20261106/vocab_scores.npz',bpe_assets=str(assets),
        window=32,carry=8,lanes=3,result_cache_entries=128,prefetch=2)
    cfg['storage'].update(project_limit_gib=1000,require_quota=False)
    cfg['model'].update(width=32,layers=2,heads=2,ffn=64)
    cfg['train'].update(batch_windows=4,micro_batch=2,log_every=1,save_every=2,milestones=[])
    provider=GroupProvider(cfg,encoder,ROOT)
    normalizer=provider.normalizer(OUT/'normalizer.json')
    source=UnitStream(provider); seen=0; units=0; blocks=set(); masks=0
    while item:=source.next_batch(4):
        seen+=item['raw_bytes']; units+=item['units']; blocks.add(item['block'])
        batch=collate(item['windows'],item['block'],normalizer)
        assert int((batch['roles']==NOISY).sum())==item['units']
        for b in range(len(item['windows'])):
            for q in torch.where(batch['roles'][b]==NOISY)[0].tolist():
                keys=torch.where(batch['allowed'][b,0,q])[0]
                for k in keys.tolist():
                    assert batch['docs'][b,q]==batch['docs'][b,k]
                    if batch['roles'][b,k] in (CLEAN,BOS):assert batch['positions'][b,k]<batch['starts'][b,q]
                    else:assert batch['noise_block'][b,k]==batch['noise_block'][b,q]
                masks+=1
    assert seen==total and blocks=={1,4,8}
    provider.close()
    net=Denoiser(cfg['model']).eval()
    # Exercise useful nonzero predictions, rather than only the zero output initialization.
    torch.nn.init.normal_(net.output.weight,std=.02)
    for block in net.blocks:torch.nn.init.normal_(block.mod.weight,std=.02)
    torch.nn.init.normal_(net.final_mod.weight,std=.02)
    schedule=lambda blocks: (1,4,8)[len(blocks)%3]
    prefix=torch.randn(3,48)
    a=generate(net,prefix,17,steps=3,seed=91,block_schedule=schedule)
    b=generate(net,prefix,17,steps=3,seed=91,block_schedule=schedule,use_cache=False)
    assert torch.allclose(a,b,atol=2e-5,rtol=2e-5)
    assert byte_lr(total,total,cfg['train'])==cfg['train']['min_lr'] or np.isclose(byte_lr(total,total,cfg['train']),cfg['train']['min_lr'])
    full=deepcopy(cfg); full['train']['run_dir']=str((OUT/'continuous_v2').relative_to(ROOT))
    split=deepcopy(cfg); split['train']['run_dir']=str((OUT/'resumed_v2').relative_to(ROOT))
    train(full,stop_after=3,device='cuda',monitor=False)
    train(split,stop_after=1,device='cuda',monitor=False)
    train(split,resume=OUT/'resumed_v2/last.pt',stop_after=2,device='cuda',monitor=False)
    x=torch.load(OUT/'continuous_v2/last.pt',map_location='cpu',weights_only=False)
    y=torch.load(OUT/'resumed_v2/last.pt',map_location='cpu',weights_only=False)
    keys=('model','ema','optimizer','scaler','sampler','rng','state','norms')
    matches={key:same(x[key],y[key]) for key in keys}
    assert all(matches.values()),matches
    evidence=dict(documents=len(documents),raw_bytes=total,units=units,blocks=sorted(blocks),
        mask_queries_checked=masks,production_packing_matches_prototype=True,cache_rebuild_vectors_exact=True,
        mixed_block_cached_full_max_error=float((a-b).abs().max()),resume_equal=matches)
    write_json(OUT/'evidence.json',evidence); print(json.dumps(evidence),flush=True)


if __name__=='__main__':main()
