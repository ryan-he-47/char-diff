"""Focused GMM likelihood, posterior/ODE, cache and warm-start recovery checks."""
import copy
import json
import math
import sys
import tempfile
import types
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from char_diff import gmm
from char_diff.data import LatentCache
from char_diff.flow import generate,loss_sum
from char_diff.model import Denoiser,NOISY
from char_diff.training import train
from char_diff.utils import ROOT,load_config,write_json


def pack(logits,means):
    return torch.cat([logits,means.flatten(-2)],-1)


def check_likelihood():
    torch.manual_seed(5)
    x=torch.randn(2,7,48)
    roles=torch.full((2,7),NOISY);roles[:,0]=0
    batch={"x":x,"roles":roles}
    mu=torch.randn_like(x,requires_grad=True)
    point,_=loss_sum(mu,batch)
    value,_=gmm.loss_sum(pack(torch.zeros(2,7,1),mu.unsqueeze(-2)),batch,.2)
    torch.testing.assert_close(point,value)
    torch.testing.assert_close(torch.autograd.grad(point,mu)[0],torch.autograd.grad(value,mu)[0])
    means=torch.randn(2,7,3,48,requires_grad=True);logits=torch.randn(2,7,3,requires_grad=True)
    raw,count=gmm.loss_sum(pack(logits,means),batch,.2)
    dist=torch.distributions.MixtureSameFamily(torch.distributions.Categorical(logits=logits),
        torch.distributions.Independent(torch.distributions.Normal(means,.2),1))
    valid=roles==NOISY
    reference=(-2*.2**2*(dist.log_prob(x)+48*math.log(.2*math.sqrt(2*math.pi))))[valid].sum()
    torch.testing.assert_close(raw,reference,atol=.0001,rtol=2e-6)
    assert count==12
    assert all(torch.isfinite(v).all() for v in torch.autograd.grad(raw,(logits,means)))
    print("PASS K=1 MSE loss/gradient and independent mixture-density likelihood",flush=True)


def exact_posterior(prior_logits,prior_means,variance,z,t):
    a=1-t;denom=a*a*variance+t*t
    means=(t*t*prior_means+variance*a*z.unsqueeze(-2))/denom
    logits=prior_logits-(z.unsqueeze(-2)-a*prior_means).square().sum(-1)/(2*denom)
    return means,logits.log_softmax(-1),variance*t*t/denom


def check_posterior():
    torch.manual_seed(7)
    prior=torch.randn(2,3,48);log_pi=torch.randn(2,3);variance=.3
    z=torch.randn(2,48);y=torch.randn_like(z)
    means,weights,v=exact_posterior(log_pi,prior,variance,z,.8)
    actual=gmm.posterior_at(pack(weights,means),z,y,.8,.3,math.sqrt(v))
    expected=exact_posterior(log_pi,prior,variance,y,.3)
    torch.testing.assert_close(actual[0],expected[0],atol=3e-6,rtol=3e-6)
    torch.testing.assert_close(actual[1],expected[1],atol=1e-5,rtol=1e-5)
    assert abs(actual[2]-expected[2])<1e-12
    pred=pack(weights,means)
    torch.testing.assert_close(gmm.ode_step(pred,z,.8,.3,.2,1),(.3/.8)*z+(1-.3/.8)*gmm.mixture_mean(pred),atol=0,rtol=0)
    # One Gaussian prior has a closed-form probability-flow trajectory.
    center=torch.linspace(-.5,.5,48).expand(2,48)
    packed=pack(torch.zeros(2,1),center.unsqueeze(-2))
    end=gmm.ode_step(packed,z,1.,0.,math.sqrt(variance),512)
    torch.testing.assert_close(end,center+math.sqrt(variance)*z,atol=.006,rtol=.006)
    assert torch.isfinite(end).all()
    print("PASS posterior ratio against Bayes rule and ODE against analytic Gaussian transport",flush=True)


def check_model_sampling():
    cfg=dict(width=64,layers=2,heads=4,ffn=128,latent_dim=48,block=16,grad_checkpoint=False)
    source=Denoiser(cfg).eval()
    with torch.no_grad():
        source.output.weight.normal_(std=.05)
        for b in source.blocks:
            b.mod.bias[128:192]=.5;b.mod.bias[320:384]=.5
    mix=Denoiser(dict(cfg,gmm_components=3,gmm_sigma=.2)).eval()
    mix.load_state_dict(gmm.expand_output_state(source.state_dict(),3,.01,20261102))
    d=LatentCache(ROOT/'data/tiny_smoke').batch([0])
    args=[d['x'],torch.zeros_like(d['positions'],dtype=torch.float32),d['roles'],d['positions'],d['lengths'],d['allowed']]
    torch.testing.assert_close(gmm.mixture_mean(mix(*args)),source(*args),atol=2e-6,rtol=2e-6)
    prefix=torch.randn(3,48)
    legacy=types.ModuleType('char_diff._pre_gmm_flow');legacy.__package__='char_diff'
    exec((ROOT/'tests/fixtures/flow_x0_reference.py').read_text(encoding='utf-8'),legacy.__dict__)
    torch.testing.assert_close(generate(source,prefix,19,3,42),legacy.generate(source,prefix,19,3,42),atol=0,rtol=0)
    for method in ['euler','gm_ode']:
        a=generate(mix,prefix,19,3,42,sampler=method)
        b=generate(mix,prefix,19,3,42,sampler=method,use_cache=False)
        torch.testing.assert_close(a,b,atol=5e-5,rtol=5e-5)
        assert torch.isfinite(a).all()
    print("PASS initial mean preservation, original sampler output and GMM cached/full generation",flush=True)


def check_training():
    cfg=load_config(ROOT/'configs/tinystories_x0_22m.yaml')
    cfg['data']['cache']='data/tiny_smoke'
    cfg['model'].update(width=64,layers=2,heads=4,ffn=128,grad_checkpoint=False)
    cfg['train'].update(batch_windows=2,micro_batch=1,precision='fp32',max_steps=4,milestones=[],save_every=99,log_every=99)
    cfg['storage'].update(require_quota=False)
    with tempfile.TemporaryDirectory(prefix='gmm_test_',dir=ROOT/'runs') as temp:
        folder=Path(temp).resolve();folder.relative_to(ROOT.resolve())
        cfg['train']['run_dir']=str(folder/'parent')
        parent=train(cfg,stop_after=2,device='cpu')['checkpoint']
        state=torch.load(parent,weights_only=False)
        lite=folder/'weights.pt'
        torch.save({k:v for k,v in state.items() if k not in ('optimizer','scaler','sampler','rng','norms')},lite)
        cache=LatentCache(ROOT/'data/tiny_smoke')
        schedule=folder/'schedule.json'
        write_json(schedule,dict(epoch_units=cache.units,total_steps=5,total_seen_units=cache.units*2,
                                 learning_rates=[.0001,.0002,.0003,.0002,.0001]))
        cfg['model'].update(gmm_components=3,gmm_sigma=.2)
        cfg['train'].update(max_steps=0,reference_schedule=str(schedule),gmm_init_rms=.01,gmm_init_seed=20261102)
        cfg['train']['run_dir']=str(folder/'full')
        full=train(cfg,gmm_from=lite,device='cpu')['checkpoint']
        cfg['train']['run_dir']=str(folder/'split')
        part=train(cfg,gmm_from=parent,stop_after=1,device='cpu')['checkpoint']
        resumed=train(cfg,resume=part,device='cpu')['checkpoint']
        a=torch.load(full,weights_only=False);b=torch.load(resumed,weights_only=False)
        assert a['state']==b['state'] and a['state']['step']==5
        assert a['sampler']==b['sampler'] and a['sampler']['cursor']==10
        for kind in ['model','ema']:
            for key in a[kind]:torch.testing.assert_close(a[kind][key],b[kind][key],atol=0,rtol=0)
        assert all(int(v['step'])==3 for v in a['optimizer']['state'].values())
        rows=[json.loads(s) for s in (folder/'full/train.jsonl').read_text().splitlines()]
        assert [r['lr'] for r in rows]==[.0003,.0002,.0001]
        changed=json.loads(schedule.read_text());changed['learning_rates'][2]=.00025;write_json(schedule,changed)
        try:train(cfg,resume=resumed,device='cpu')
        except ValueError as e:assert 'schedule differs' in str(e)
        else:raise AssertionError('Changed reference schedule accepted')
    print("PASS identical full/weight-only warm-start, exact recovery, global schedule and common endpoint",flush=True)


if __name__=='__main__':
    torch.set_num_threads(2)
    check_likelihood();check_posterior();check_model_sampling();check_training()
    write_json(ROOT/'docs/experiments/gmm-switch-timing/evidence/targeted-tests.json',
               dict(date='2026-11-02',passed=['likelihood_gradient','posterior_ode','sampling_cache','warm_start_recovery_schedule'],torch=torch.__version__))
