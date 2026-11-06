"""Targeted checks: shifted AR data, causal/KV path, legacy likelihood and AMP."""
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from char_diff.ar import ARCache, GMMAR, history_batch, readout
from char_diff.data import LatentCache
from char_diff.model import NOISY
from char_diff.utils import ROOT, load_config, write_json


def main():
    torch.set_num_threads(4); torch.manual_seed(1234)
    cache = ARCache(ROOT/'data/tinystories')
    base = LatentCache(ROOT/'data/tinystories')
    # One ordinary window, one multi-story window, one carried-history window.
    multi = int(np.flatnonzero(np.diff(cache.windows) > 1)[0])
    seg = int(np.flatnonzero(cache.segments[:,1] > cache.segments[:,3])[0])
    carried = int(np.searchsorted(cache.windows, seg, side='right')-1)
    indices = [0, multi, carried]
    batch = cache.batch(indices, 'cuda'); old = base.batch(indices, 'cuda')
    for i in range(len(indices)):
        torch.testing.assert_close(batch['target'][i,batch['mask'][i]], old['x'][i,old['roles'][i] == NOISY], rtol=0, atol=0)
    assert int(batch['mask'].sum()) == cache.target_units(indices)
    for i in range(len(indices)):
        for j in range(batch['x'].shape[1]):
            if batch['docs'][i,j] < 0: continue
            if batch['bos'][i,j]: assert not batch['x'][i,j].any()
            else: torch.testing.assert_close(batch['x'][i,j], batch['target'][i,j-1], rtol=0, atol=0)
            keys = batch['allowed'][i,0,j].nonzero().flatten()
            assert (batch['docs'][i,keys] == batch['docs'][i,j]).all()
            assert (batch['positions'][i,keys] <= batch['positions'][i,j]).all()
    cfg = load_config(ROOT/'configs/tinystories_gmm_ar_22m.yaml')['model']
    model = GMMAR(cfg).cuda().eval()
    parameters = sum(p.numel() for p in model.parameters())
    h = torch.randn(5, cfg['width'], device='cuda')
    y = torch.randn(5, 48, device='cuda')
    logits, mu, sigma = model.head.params(h)
    mix = torch.distributions.MixtureSameFamily(torch.distributions.Categorical(logits=logits),
            torch.distributions.Independent(torch.distributions.Normal(mu, sigma), 1))
    torch.testing.assert_close(model.head.nll(h,y), -mix.log_prob(y), atol=3e-5, rtol=1e-5)
    with torch.no_grad():
        saved = model.head.out.bias.clone()
        model.head.out.bias[16+16*48:] = -100
        _,_,floor = model.head.params(torch.zeros_like(h))
        torch.testing.assert_close(floor, torch.full_like(floor,.05), atol=0, rtol=0)
        model.head.out.bias.copy_(saved)
    with torch.inference_mode():
        histories = [cache.story(0)[:7], cache.story(1)[:4]]
        packed, lengths = history_batch(histories, 'cuda')
        full = model(**packed)
        prefix, _ = history_batch([histories[0][:3]], 'cuda')
        hv, kv = model(**prefix, return_cache=True)
        errors = []
        for j in range(3, 7):
            p = torch.tensor([[j+1]], device='cuda')
            hv,kv = model(torch.tensor(histories[0][j:j+1],device='cuda')[:,None], p,
                torch.zeros_like(p,dtype=torch.bool), torch.ones(1,1,1,j+2,device='cuda',dtype=torch.bool),
                past=kv, return_cache=True)
            errors.append(float((hv[0,0]-full[0,j+1]).abs().max()))
            torch.testing.assert_close(hv[0,0],full[0,j+1], atol=3e-5,rtol=3e-5)
        changed = dict(packed); changed['x'] = packed['x'].clone(); changed['x'][:,5:] += 3
        torch.testing.assert_close(model(**changed)[:,:5],full[:,:5],atol=0,rtol=0)
        # Packed histories have the same outputs as independent, unpadded contexts.
        single,_ = history_batch([histories[1]],'cuda')
        torch.testing.assert_close(model(**single)[0,-1], full[1,lengths[1]-1], atol=3e-5,rtol=3e-5)
        peaks = readout(model,h[:2],torch.Generator(device='cuda').manual_seed(42))
        assert peaks.shape == (2,48) and torch.isfinite(peaks).all()
    model.train()
    micro = cache.batch([0], 'cuda')
    with torch.autocast('cuda', dtype=torch.float16):
        raw,n,stats = model.loss_sum(micro,torch.tensor(cache.mean,device='cuda'),cache.scale,True)
        assert raw.dtype == torch.float32
    (raw/n).backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    report = {'parameters': parameters,'target_units': int(batch['mask'].sum()), 'windows':indices,
              'kv_max_abs':max(errors),'fp16_nll':float(raw.detach()/n),'diagnostics':stats,
              'checks':['identical targets and story-isolated shifted inputs','causal/padded/full vs KV',
                        'legacy raw sigma floor and analytic NLL','22M FP16 backward, FP32 head, finite density readout']}
    write_json(ROOT/'runs/gmm_ar_build_20261103/targeted-checks.json',report)
    print(json.dumps(report),flush=True)


if __name__ == '__main__': main()
