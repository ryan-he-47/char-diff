"""Paired noise draws, full-state continuation and exact branch recovery."""
import copy
import json
import sys
import tempfile
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from char_diff.data import LatentCache
from char_diff.flow import corrupt
from char_diff.training import train
from char_diff.utils import ROOT,load_config,sha256,write_json


def main():
    torch.set_num_threads(2)
    cache=LatentCache(ROOT/"data/tiny_smoke");batch=cache.batch([0])
    a=torch.Generator().manual_seed(123);b=torch.Generator().manual_seed(123)
    x,t=corrupt(batch,a);y,u=corrupt(batch,b,time_sampling="beta_2_1")
    valid=batch["noise_block"]>=0
    torch.testing.assert_close(u[valid].square(),t[valid],atol=1e-7,rtol=1e-7)
    assert torch.equal(a.get_state(),b.get_state())
    golden=torch.Generator().manual_seed(123)
    times=torch.rand(int(batch["noise_block"].max())+1,generator=golden)
    expected=torch.where(valid,times[batch["noise_block"].clamp_min(0)],0.0)
    eps=torch.randn(batch["x"].shape,generator=golden)
    torch.testing.assert_close(t,expected,atol=0,rtol=0)
    torch.testing.assert_close(x,(1-t[...,None])*batch["x"]+t[...,None]*eps,atol=0,rtol=0)
    fixed_a=corrupt(batch,torch.Generator().manual_seed(1),.75)
    fixed_b=corrupt(batch,torch.Generator().manual_seed(1),.75,"beta_2_1")
    for first,second in zip(fixed_a,fixed_b):torch.testing.assert_close(first,second,atol=0,rtol=0)
    print("PASS legacy uniform corruption, paired random draws and fixed-time evaluation",flush=True)

    cfg=load_config(ROOT/"configs/tinystories_x0_22m.yaml")
    cfg["data"]["cache"]="data/tiny_smoke"
    cfg["model"].update(width=64,layers=2,heads=4,ffn=128,grad_checkpoint=False)
    cfg["train"].update(batch_windows=2,micro_batch=1,precision="fp32",max_steps=4,milestones=[],save_every=1,log_every=99)
    with tempfile.TemporaryDirectory(prefix="continuation_",dir=ROOT/"runs") as temp:
        folder=Path(temp).resolve();folder.relative_to(ROOT.resolve())
        def config(name,distribution="uniform"):
            c=copy.deepcopy(cfg);c["train"].update(run_dir=str(folder/name),time_sampling=distribution);return c
        parent=train(config("parent"),stop_after=2,device="cpu")["checkpoint"];parent_sha=sha256(parent)
        full=train(config("full"),fork_from=parent,stop_after=2,device="cpu")["checkpoint"]
        interrupted=train(config("interrupted"),fork_from=parent,stop_after=1,device="cpu")["checkpoint"]
        resumed=train(config("interrupted"),resume=interrupted,stop_after=1,device="cpu")["checkpoint"]
        beta=train(config("beta","beta_2_1"),fork_from=parent,stop_after=2,device="cpu")["checkpoint"]
        p=torch.load(parent,weights_only=False);f=torch.load(full,weights_only=False);r=torch.load(resumed,weights_only=False);h=torch.load(beta,weights_only=False)
        assert sha256(parent)==parent_sha
        assert f["state"]==r["state"] and f["sampler"]==r["sampler"]==h["sampler"]
        for arm in (f,r,h):
            assert arm["state"]["step"]==2 and arm["lineage"]["source_step"]==2
            assert arm["lineage"]["source_sha256"]==parent_sha
            assert all(int(v["step"])==4 for v in arm["optimizer"]["state"].values())
        for kind in ("model","ema"):
            for key in f[kind]:torch.testing.assert_close(f[kind][key],r[kind][key],atol=0,rtol=0)
        assert torch.equal(f["rng"]["torch"],r["rng"]["torch"])
        assert torch.equal(f["rng"]["torch"],h["rng"]["torch"])
        assert any(not torch.equal(f["model"][k],h["model"][k]) for k in f["model"])
        changed=config("full","beta_2_1")
        try:train(changed,resume=full,device="cpu")
        except ValueError as e:assert "time_sampling differs" in str(e)
        else:raise AssertionError("Exact resume accepted a different time distribution")
    print("PASS immutable parent, retained optimizer steps, paired cursor/RNG and exact branch recovery",flush=True)
    write_json(ROOT/"docs/experiments/high-noise-continuation/evidence/continuation-tests.json",{"date":"2026-11-01","passed":["legacy_uniform_corruption","paired_draw_layout","fixed_time_evaluation","immutable_parent","full_state_fork","exact_branch_resume","distribution_resume_guard"],"torch":torch.__version__})


if __name__=="__main__":main()
