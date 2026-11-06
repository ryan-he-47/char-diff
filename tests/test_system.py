"""Causal isolation, cached sampling, corpus coverage and exact resume checks."""
import copy
import json
import sys
import tempfile
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from char_diff.data import LatentCache,WindowSampler
from char_diff.model import Denoiser,CLEAN,NOISY
from char_diff.flow import corrupt,generate,loss_sum
from char_diff.training import train
from char_diff.storage import StorageGuard
from char_diff.utils import ROOT,load_config,write_json
from char_diff.cache import prepare
from scripts.verify_cache import verify

def active_model():
    torch.manual_seed(7)
    model=Denoiser(dict(width=64,layers=2,heads=4,ffn=128,latent_dim=48,block=16,grad_checkpoint=False)).eval()
    with torch.no_grad():
        torch.nn.init.normal_(model.output.weight,std=0.05)
        for b in model.blocks:
            b.mod.bias[2*64:3*64]=1;b.mod.bias[5*64:6*64]=1
    return model

def main():
    torch.set_num_threads(2)
    cache=LatentCache(ROOT/"data/tiny_smoke",verify_hashes=True)
    dev=LatentCache(ROOT/"data/tiny_smoke","dev")
    assert sum(int(x) for x in cache.valid)==cache.units
    counts=np.fromfile(cache.root/"train_counts.u64",dtype="<u8").astype(np.float64)
    assert np.max(np.abs((cache.vectors*counts[:,None]).sum(0)/counts.sum()))<1e-5
    assert abs((cache.vectors.astype(np.float64)**2*counts[:,None]).sum()/(48*counts.sum())-1)<1e-5
    print("PASS cache integrity, coverage and frequency-weighted normalization",flush=True)

    idx=next(i for i in range(len(dev)) if dev.windows[i+1]-dev.windows[i]>1)
    batch=dev.batch([idx]);model=active_model()
    generator=torch.Generator().manual_seed(123);x,t=corrupt(batch,generator)
    call=lambda value:model(value,t,batch["roles"],batch["positions"],batch["lengths"],batch["allowed"])
    with torch.inference_mode():
        original=call(x)
        q=(batch["roles"]==NOISY)&(batch["docs"]==0)&(batch["noise_block"]==0)
        altered=x.clone();future=(batch["roles"]==CLEAN)&(batch["docs"]==0)
        altered[future]+=torch.randn_like(altered[future])*20
        torch.testing.assert_close(call(altered)[q],original[q],atol=1e-6,rtol=1e-6)
        changed=x.clone();other=batch["docs"]==1;changed[other]+=torch.randn_like(changed[other])*20
        doc0=(batch["docs"]==0)&(batch["roles"]==NOISY)
        torch.testing.assert_close(call(changed)[doc0],original[doc0],atol=1e-6,rtol=1e-6)
        later=(batch["roles"]==NOISY)&(batch["docs"]==0)&(batch["noise_block"]==1)
        assert (call(altered)[later]-original[later]).abs().max()>1e-4
    print("PASS future isolation, document isolation and positive history sensitivity",flush=True)

    prefix=torch.randn(13,48)
    a=generate(model,prefix,35,steps=3,seed=88,use_cache=True)
    b=generate(model,prefix,35,steps=3,seed=88,use_cache=False)
    torch.testing.assert_close(a,b,atol=3e-5,rtol=3e-5)
    print("PASS KV-cache sampling equals full recomputation (including a partial block)",flush=True)

    identity=generate(model,prefix,35,steps=3,seed=88,history_transform=lambda value:value)
    torch.testing.assert_close(identity,a,atol=0,rtol=0)
    shapes=[]
    def shifted(value):
        shapes.append(tuple(value.shape))
        return value.add_(0.25)
    projected=generate(model,prefix,35,steps=3,seed=88,history_transform=shifted)
    assert shapes==[(16,48),(16,48),(3,48)]
    torch.testing.assert_close(projected[:16],a[:16],atol=0,rtol=0)
    assert (projected[16:]-a[16:]).abs().max()>1e-5
    recomputed=generate(model,prefix,35,steps=3,seed=88,use_cache=False,history_transform=lambda value:value+0.25)
    torch.testing.assert_close(projected,recomputed,atol=3e-5,rtol=3e-5)
    print("PASS identity history, first-block emission isolation and transformed KV-cache equivalence",flush=True)

    model.train();model.cfg["grad_checkpoint"]=True
    reference=copy.deepcopy(model);reference.cfg["grad_checkpoint"]=False
    raw,count=loss_sum(model(x,t,batch["roles"],batch["positions"],batch["lengths"],batch["allowed"]),batch)
    (raw/(48*count)).backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    raw_ref,count_ref=loss_sum(reference(x,t,batch["roles"],batch["positions"],batch["lengths"],batch["allowed"]),batch)
    (raw_ref/(48*count_ref)).backward()
    for a,b in zip(model.parameters(),reference.parameters()):
        torch.testing.assert_close(a.grad,b.grad,atol=1e-7,rtol=1e-5)
    print("PASS checkpointed backward equals ordinary backward",flush=True)

    assert torch.all(t[batch["roles"]!=NOISY]==0)
    for bid in batch["noise_block"][batch["noise_block"]>=0].unique():
        assert t[batch["noise_block"]==bid].unique().numel()==1
    print("PASS noise time is shared within a block and clean history stays at t=0",flush=True)

    try:StorageGuard(ROOT,limit_gib=0,reserve_gib=0).check(1)
    except RuntimeError:pass
    else:raise AssertionError("Storage guard failed to close admission")
    print("PASS storage budget admission",flush=True)

    (ROOT/"runs").mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="test_resume_",dir=ROOT/"runs") as folder:
        root=Path(folder).resolve();root.relative_to(ROOT.resolve())
        cfg=load_config(ROOT/"configs/tinystories_x0_22m.yaml")
        cfg["model"].update(width=64,layers=2,heads=4,ffn=128,grad_checkpoint=False)
        cfg["data"]["cache"]="data/tiny_smoke"
        cfg["train"].update(max_steps=6,batch_windows=2,micro_batch=1,precision="fp32",log_every=6,save_every=3,milestones=[])
        cfg["train"]["run_dir"]=str(root/"continuous")
        first=train(copy.deepcopy(cfg),device="cpu")
        full=torch.load(first["checkpoint"],map_location="cpu",weights_only=False)
        previous=torch.load(Path(first["checkpoint"]).with_name("previous.pt"),map_location="cpu",weights_only=False)
        assert previous["state"]["step"]==3
        cfg["train"]["run_dir"]=str(root/"resumed")
        half=train(copy.deepcopy(cfg),stop_after=3,device="cpu")
        second=train(copy.deepcopy(cfg),resume=half["checkpoint"],device="cpu")
        resumed=torch.load(second["checkpoint"],map_location="cpu",weights_only=False)
        assert full["state"]==resumed["state"]
        for kind in ["model","ema"]:
            for key in full[kind]:torch.testing.assert_close(full[kind][key],resumed[kind][key],rtol=0,atol=0)
        assert full["sampler"]==resumed["sampler"]
        changed=copy.deepcopy(cfg);changed["train"]["ema"]=0.9
        try:train(changed,resume=second["checkpoint"],device="cpu")
        except ValueError as e:assert "train.ema differs" in str(e)
        else:raise AssertionError("Changed resume configuration was accepted")

        # A small deterministic fixture exercises cleaning and source provenance.
        class FakeCodec:
            device="cpu"
            fingerprint={"test_fixture":True}
            verifier=type("Stats",(),{"n_flag":0,"n_floor_fail":0})()
            def split(self,texts):return [[text.encode()] for text in texts]
            def encode(self,units):return np.array([[u[0]*(i+1)/48 for i in range(48)] for u in units],np.float32)
        src_train=root/"train.jsonl";src_val=root/"val.jsonl"
        for path,texts in [(src_train,["A","A","","shared","B"]),(src_val,["shared","shared","z"])]:
            path.write_text("".join(json.dumps({"text":text})+"\n" for text in texts),encoding="utf-8")
        manifest=prepare(root/"fixture",FakeCodec(),StorageGuard(ROOT),src_train,src_val)
        assert manifest["splits"]["train"]["stories"]==2
        assert manifest["cleaning"]["train_duplicates"]==1
        assert manifest["cleaning"]["train_empty"]==1
        assert manifest["cleaning"]["train_validation_overlap"]==1
        assert manifest["cleaning"]["validation_duplicates"]==1
        assert "sha256" in manifest["sources"]["train_jsonl"]
        verify(root/"fixture")
        assert not (root/"fixture.building").exists()
    print("PASS interrupted/resumed run matches uninterrupted weights, EMA and data cursor exactly",flush=True)
    print("PASS resume configuration guard and deduplicated cache construction",flush=True)
    write_json(ROOT/"docs/experiments/infrastructure/evidence/implementation-tests.json",{"passed":["cache_integrity","corpus_coverage","weighted_normalization","future_isolation","document_isolation","history_sensitivity","kv_cache_equivalence","checkpointed_backward_equivalence","per_block_noise_time","storage_admission","exact_resume","resume_config_guard","checkpoint_rotation","deduplicated_cache_build","identity_history_equivalence","first_block_emission_isolation","transformed_kv_cache_equivalence"],"torch":torch.__version__})

if __name__=="__main__":main()
