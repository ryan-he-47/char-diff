"""Targeted checks for velocity loss, ODE direction, legacy sampling and target recovery."""
import copy
import json
import sys
import tempfile
import types
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from char_diff.data import LatentCache
from char_diff.flow import corrupt,generate,loss_sum,prediction_to_data
from char_diff.model import Denoiser
from char_diff.training import train
from char_diff.utils import ROOT,load_config,write_json


def main():
    torch.set_num_threads(2)
    batch=LatentCache(ROOT/"data/tiny_smoke").batch([0])
    x,t,eps=corrupt(batch,torch.Generator().manual_seed(42),fixed_t=.25,return_noise=True)
    pred=torch.randn_like(x,requires_grad=True)
    velocity_target=eps-batch["x"]
    native,n=loss_sum(pred,batch,target=velocity_target)
    weighted,_=loss_sum(pred,batch,target=velocity_target,weights=t.square())
    converted,_=loss_sum(prediction_to_data(pred,x,t,"v"),batch)
    torch.testing.assert_close(weighted,converted,atol=.001,rtol=2e-6)
    a=torch.autograd.grad(weighted,pred,retain_graph=True)[0]
    b=torch.autograd.grad(converted,pred,retain_graph=True)[0]
    torch.testing.assert_close(a,b,atol=1e-6,rtol=2e-5)
    assert native>0 and n>0 and torch.isfinite(torch.autograd.grad(native,pred)[0]).all()
    for time in [0.,1e-7,1.]:
        actual=prediction_to_data(velocity_target,x,torch.full_like(t,time),"v")
        assert torch.isfinite(actual).all()
    print("PASS v=eps-x, native loss, x0-equivalent weighted loss and gradients",flush=True)

    class ConstantVelocity(torch.nn.Module):
        def __init__(self):
            super().__init__();self.velocity=torch.nn.Parameter(torch.linspace(-.5,.5,48))
            self.cfg={"latent_dim":48,"block":16}
        def forward(self,z,*args,return_cache=False,**kwargs):
            value=self.velocity.expand_as(z)
            return (value,None) if return_cache else value
    oracle=ConstantVelocity();rng=torch.Generator().manual_seed(88)
    expected=torch.cat([torch.randn(1,k,48,generator=rng)[0]-oracle.velocity for k in [16,16,3]])
    for steps in [1,7]:
        for cached in [False,True]:
            value=generate(oracle,torch.empty(0,48),35,steps,88,use_cache=cached,prediction_type="v")
            torch.testing.assert_close(value,expected,atol=2e-6,rtol=2e-6)
    print("PASS velocity integration sign and endpoint across blocks, partial blocks and step counts",flush=True)

    torch.manual_seed(7)
    model=Denoiser(dict(width=64,layers=2,heads=4,ffn=128,latent_dim=48,block=16,grad_checkpoint=False)).eval()
    with torch.no_grad():
        torch.nn.init.normal_(model.output.weight,std=.05)
        for block in model.blocks:block.mod.bias[128:192]=1;block.mod.bias[320:384]=1
    prefix=torch.randn(13,48)
    legacy=types.ModuleType("char_diff._pre_v_flow");legacy.__package__="char_diff"
    exec((Path(__file__).resolve().parent/"fixtures/flow_x0_reference.py").read_text(encoding="utf-8"),legacy.__dict__)
    torch.testing.assert_close(generate(model,prefix,35,3,88),legacy.generate(model,prefix,35,3,88),atol=0,rtol=0)
    a=generate(model,prefix,35,3,88,prediction_type="v")
    b=generate(model,prefix,35,3,88,use_cache=False,prediction_type="v")
    torch.testing.assert_close(a,b,atol=3e-5,rtol=3e-5)
    print("PASS legacy x0 output unchanged and velocity KV-cache/full-recompute agreement",flush=True)

    cfg=load_config(ROOT/"configs/tinystories_x0_22m.yaml")
    cfg["data"]["cache"]="data/tiny_smoke"
    cfg["model"].update(width=64,layers=2,heads=4,ffn=128,grad_checkpoint=False)
    cfg["train"].update(prediction_type="v",loss_space="x0",batch_windows=2,micro_batch=1,precision="fp32",max_steps=3,milestones=[],log_every=99,save_every=99)
    with tempfile.TemporaryDirectory(prefix="prediction_",dir=ROOT/"runs") as temp:
        folder=Path(temp).resolve();folder.relative_to(ROOT.resolve())
        c=copy.deepcopy(cfg);c["train"]["run_dir"]=str(folder/"full")
        full=train(c,device="cpu")["checkpoint"]
        c["train"]["run_dir"]=str(folder/"split")
        part=train(c,stop_after=1,device="cpu")["checkpoint"]
        resumed=train(c,resume=part,device="cpu")["checkpoint"]
        f=torch.load(full,weights_only=False);r=torch.load(resumed,weights_only=False)
        assert f["state"]==r["state"] and f["sampler"]==r["sampler"]
        for kind in ["model","ema"]:
            for key in f[kind]:torch.testing.assert_close(f[kind][key],r[kind][key],atol=0,rtol=0)
        for field,value in [("prediction_type","x0"),("loss_space","native")]:
            changed=copy.deepcopy(c);changed["train"][field]=value
            try:train(changed,resume=resumed,device="cpu")
            except ValueError as e:assert field in str(e)
            else:raise AssertionError("Changed objective was accepted for exact resume")
    print("PASS velocity training recovery and prediction/loss-space guards",flush=True)
    write_json(ROOT/"docs/experiments/prediction-target/evidence/targeted-tests.json",{"date":"2026-11-01","passed":["velocity_loss_and_gradients","velocity_ode_direction_and_endpoint","legacy_x0_and_velocity_kv_sampling","velocity_exact_recovery_and_target_guards"],"torch":torch.__version__})


if __name__=="__main__":main()
