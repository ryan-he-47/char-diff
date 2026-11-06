"""Targeted checks for the new B1 selection/RNG/KV path only."""
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from char_diff.codec import Codec
from char_diff.data import LatentCache
from char_diff.flow import generate
from char_diff.model import Denoiser
from char_diff.selection import CycleScorer, CycleRetry, prefer_retry
from char_diff.utils import ROOT, write_json


class ForcedRetry:
    def __init__(self, accept): self.accept = accept; self.selected = []; self.calls = 0
    def __call__(self, first, retry):
        other = retry(); self.calls += 1
        selected = other if self.accept else first
        self.selected.append(selected.clone())
        return selected


class TraceKV(torch.nn.Module):
    def __init__(self, base):
        super().__init__(); self.base = base; self.cfg = base.cfg; self.committed = []
    def forward(self, x, *args, **kwargs):
        if kwargs.get('return_cache') and kwargs.get('past') is not None: self.committed.append(x[0].clone())
        return self.base(x, *args, **kwargs)


@torch.inference_mode()
def main():
    torch.set_num_threads(4)
    folder = ROOT / 'runs/cycle_retry_20261102'
    state = torch.load(ROOT / 'runs/tinystories_x0_block1/last.pt', map_location='cpu', weights_only=False)
    model = Denoiser(state['config']['model']).cuda().eval(); model.load_state_dict(state['ema'])
    codec = Codec(device='cuda'); cache = LatentCache(ROOT / 'data/tinystories', 'dev')
    prefix = cache.story(6501)[:32]
    archived = np.load(ROOT / 'runs/block_size_review_20261102/sample-latents.npz')
    old = torch.tensor(archived['b1_doc6501_seed42'][:8], device='cuda')
    with torch.autocast('cuda', dtype=torch.float16): actual = generate(model, prefix, 8, 32, 42)
    assert torch.equal(actual, old), 'Default sampling must reproduce saved FP16 latents'
    scorer = CycleScorer(codec, state['normalizer'])
    plain = scorer(old)
    with torch.autocast('cuda', dtype=torch.float16): under_amp = scorer(old)
    assert plain == under_amp
    previous = [json.loads(s) for s in (ROOT / 'runs/decoder_friendliness_probe_20261102/units.jsonl').read_text(encoding='utf-8').splitlines()][:8]
    errors = [abs(a['cycle_rmse'] - b['cycle_rmse_standardized']) for a, b in zip(plain, previous)]
    assert max(errors) < 2e-6 and [a['unit_hex'] for a in plain] == [b['hex'] for b in previous]
    base = generate(model, prefix, 3, 2, 77)
    keep = ForcedRetry(False)
    retained = generate(model, prefix, 3, 2, 77, unit_selector=keep)
    assert keep.calls == 3 and torch.equal(base, retained)
    take = ForcedRetry(True); traced = TraceKV(model)
    cached = generate(traced, prefix, 3, 2, 77, unit_selector=take)
    direct = generate(model, prefix, 3, 2, 77, use_cache=False, unit_selector=ForcedRetry(True))
    torch.testing.assert_close(cached, direct, atol=3e-4, rtol=3e-4)
    assert len(traced.committed) == 3
    assert all(torch.equal(a, b) for a, b in zip(traced.committed, take.selected))
    assert torch.equal(cached, torch.cat(take.selected))
    # Exercise the real selector with two existing valid candidates, no extra diffusion.
    all_z = torch.tensor(np.concatenate([archived[k] for k in archived.files]), device='cuda')
    scored = scorer(all_z)
    good = min(range(len(scored)), key=lambda i: scored[i]['cycle_rmse'])
    bad = max((i for i in range(len(scored)) if scored[i]['valid']), key=lambda i: scored[i]['cycle_rmse'])
    selector = CycleRetry(scorer, .1)
    chosen = selector(all_z[bad:bad+1], lambda: all_z[good:good+1])
    assert torch.equal(chosen, all_z[good:good+1]) and selector.summary()['retries'] == 1
    chosen = selector(all_z[good:good+1], lambda: (_ for _ in ()).throw(AssertionError('Unexpected retry')))
    assert torch.equal(chosen, all_z[good:good+1])
    assert not prefer_retry(scored[good], scored[bad])
    assert not prefer_retry(scored[good], scored[good])
    result = {'default_fp16_latents_exact': 8, 'scorer_amp_equal_fp32': True,
              'max_cycle_difference_vs_probe': max(errors), 'original_rng_unchanged_with_discarded_retries': True,
              'selected_only_kv_commits': len(traced.committed),
              'cached_vs_full_history_max_abs': float((cached-direct).abs().max()),
              'real_selector_trigger_accept_and_no_retry': selector.summary()}
    write_json(folder / 'targeted-checks.json', result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__': main()
