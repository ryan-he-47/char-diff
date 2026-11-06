"""Final GMM training summary, fixed dev buckets, and a small paired text panel."""
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from char_diff.codec import Codec
from char_diff.data import LatentCache
from char_diff.flow import corrupt, generate, loss_sum, prediction_to_data
from char_diff.gmm import loss_sum as gmm_loss_sum, diagnostics
from char_diff.model import Denoiser
from char_diff.utils import ROOT, sha256, write_json


@torch.inference_mode()
def main():
    torch.set_num_threads(4)
    folder = ROOT / "runs/gmm_review_20261102"
    cache = LatentCache(ROOT / "data/tinystories", "dev")
    codec = Codec(device="cuda")
    mean = torch.tensor(cache.mean, device="cuda")
    restart = json.loads((ROOT / "docs/experiments/gmm-switch-timing/evidence/recovery-submission.json").read_text())
    old = json.loads((ROOT / "runs/continuation_review_20261101/summary.json").read_text())
    panel = json.loads((ROOT / "runs/review_715121/panel/prompts.json").read_text(encoding='utf-8'))['prompts'][:4]
    controls = [json.loads(s) for s in (ROOT / "runs/continuation_review_20261101/samples.jsonl").read_text(encoding="utf-8").splitlines()]
    report = {"control_dev_ema": old['validation']['uniform']['ema'], "arms": {},
              "protocol": {"dev_windows": 64, "times": [.05, .25, .5, .75, .95], "noise_seed": 31415,
                           "evaluation_precision": "fp32", "weights": "ema", "sample_prompts": [q['doc'] for q in panel],
                           "sample_seed": 42, "sample_units": 128, "steps": 32, "sample_precision": "fp16 with fp32 GMM residuals"}}
    records = []; vectors = {}
    with (folder / 'samples.jsonl').open('x', encoding='utf-8', buffering=1) as log:
        for arm in ('early', 'late'):
            path = folder / arm / 'last.pt'
            state = torch.load(path, map_location='cpu', weights_only=False)
            assert state['state']['step'] == 28989 and state['cache_fingerprint'] == cache.fingerprint
            assert state['codec'] == codec.fingerprint
            rows = [json.loads(s) for s in (folder / arm / 'train.jsonl').read_text().splitlines()]
            cut = restart['pre_restart_train_log_rows'][arm]; start = restart['approved_resume_steps'][arm]
            trajectory = [r for r in rows[:cut] if r['updated'] and r['step'] <= start] + [r for r in rows[cut:] if r['updated']]
            assert len(trajectory) == 28989 - state['lineage']['source_step']
            good = [r for r in trajectory if math.isfinite(r['loss'])]
            model = Denoiser(state['config']['model']).cuda().eval()
            model.load_state_dict(state['ema'])
            result = {'checkpoint': str(path), 'sha256': sha256(path), 'state': state['state'],
                      'source_step': state['lineage']['source_step'], 'retained_gmm_updates': len(trajectory),
                      'resumed_successful_updates': len([r for r in rows[cut:] if r['updated']]),
                      'resumed_skips': len([r for r in rows[cut:] if not r['updated']]),
                      'first200_loss': float(np.mean([r['loss'] for r in good[:200]])),
                      'last200_loss': float(np.mean([r['loss'] for r in good[-200:]])), 'dev_ema': []}
            for t in report['protocol']['times']:
                rng = torch.Generator(device='cuda').manual_seed(31415)
                total = nll = count = 0
                probe = None
                for i in range(64):
                    batch = cache.batch([i], 'cuda'); x, times = corrupt(batch, rng, t)
                    prediction = model(x, times, batch['roles'], batch['positions'], batch['lengths'], batch['allowed'])
                    value, n = loss_sum(prediction_to_data(prediction, x, times), batch)
                    density, _ = gmm_loss_sum(prediction, batch, .2)
                    total += float(value); nll += float(density); count += n
                    if i == 0: probe = diagnostics(prediction, batch, .2)
                result['dev_ema'].append({'time': t, 'mse': total / (48 * count),
                                          'scaled_gmm_loss': nll / (48 * count), 'units': count, 'probe': probe})
            report['arms'][arm] = result
            write_json(folder / 'review-progress.json', report)
            print(json.dumps({'arm': arm, 'last200_loss': result['last200_loss'], 'resumed_skips': result['resumed_skips'],
                              'dev_mse': [r['mse'] for r in result['dev_ema']]}), flush=True)
            for q in panel:
                for sampler in ('euler', 'gm_ode'):
                    tick = time.perf_counter()
                    with torch.autocast('cuda', dtype=torch.float16):
                        z = generate(model, cache.story(q['doc'])[:32], 128, 32, 42, sampler=sampler, substeps=4)
                    assert torch.isfinite(z).all()
                    units = codec.decode(z * cache.scale + mean); raw = b''.join(units)
                    record = {'arm': arm, 'sampler': sampler, 'doc': q['doc'], 'seed': 42, 'prompt': q['prompt'],
                              'generated': raw.decode('utf-8', errors='replace'), 'unit_hex': [u.hex() for u in units],
                              'utf8_valid': raw.decode('utf-8', errors='replace').encode('utf-8') == raw,
                              'seconds': time.perf_counter() - tick}
                    records.append(record); log.write(json.dumps(record, ensure_ascii=False) + '\n')
                    vectors[f"{arm}_{sampler}_{q['doc']}"] = z.cpu().numpy()
                    print(json.dumps({'sample': [arm, sampler, q['doc']], 'seconds': record['seconds']}), flush=True)
            del state, model
    report['sample_cases'] = len(records); report['utf8_valid_cases'] = sum(r['utf8_valid'] for r in records)
    write_json(folder / 'summary.json', report)
    write_json(ROOT / 'docs/experiments/gmm-switch-timing/evidence/final-review.json', report)
    np.savez(folder / 'sample_latents.npz', **vectors)
    lines = ['# GMM终点与已有x₀控制：固定前缀短续写', '']
    for q in panel:
        lines += [f"## doc {q['doc']} / seed42", '', '**前缀**', '', q['prompt'], '']
        control = next(r for r in controls if r['arm'] == 'uniform' and r['doc'] == q['doc'] and r['seed'] == 42)
        lines += ['**x₀控制717047（已缓存）**', '', control['generated'], '']
        for r in records:
            if r['doc'] == q['doc']: lines += [f"**{r['arm']} / {r['sampler']}**", '', r['generated'], '']
    (folder / 'samples.md').write_text('\n'.join(lines), encoding='utf-8')


if __name__ == '__main__':
    main()
