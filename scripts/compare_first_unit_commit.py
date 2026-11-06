"""Paired text preview: denoise 16 units, commit one, then redraw a full block."""
import json
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from char_diff.codec import Codec
from char_diff.data import LatentCache
from char_diff.flow import generate
from char_diff.model import Denoiser, NOISY
from char_diff.utils import ROOT, sha256, write_json


class CommitTrace(torch.nn.Module):
    def __init__(self, model, check=False, doc=None):
        super().__init__(); self.base = model; self.cfg = model.cfg
        self.check = check; self.doc = doc; self.calls = 0; self.committed = 0; self.cache_lengths = []

    def forward(self, x, t, roles, positions, lengths, allowed, **kwargs):
        clean = kwargs.get('return_cache', False)
        if not clean:
            self.calls += 1
            if self.check:
                assert x.shape[1] == 16 and (roles == NOISY).all() and (lengths == 16).all()
                assert positions[0, 0] == 33 + self.committed
        result = self.base(x, t, roles, positions, lengths, allowed, **kwargs)
        if clean and kwargs.get('past') is not None:
            assert x.shape[1] == 1
            self.committed += 1
            size = result[1][0][0].shape[-2]
            assert size == 33 + self.committed
            self.cache_lengths.append(size)
            if self.doc is not None and self.committed % 16 == 0:
                print(json.dumps({'doc': self.doc, 'committed_units': self.committed, 'denoiser_calls': self.calls}), flush=True)
        return result


@torch.inference_mode()
def main():
    torch.set_num_threads(4)
    folder = ROOT / 'runs/first_unit_commit_20261102'; folder.mkdir(parents=True, exist_ok=True)
    if (folder / 'pairs.jsonl').exists(): raise FileExistsError('Preserve prior run; choose a new output directory')
    path = ROOT / 'runs/continuation_review_20261101/uniform/last.pt'
    state = torch.load(path, map_location='cpu', weights_only=False)
    cache = LatentCache(ROOT / 'data/tinystories', 'dev'); codec = Codec(device='cuda')
    assert state['cache_fingerprint'] == cache.fingerprint and state['codec'] == codec.fingerprint
    model = Denoiser(state['config']['model']).cuda().eval(); model.load_state_dict(state['ema'])
    prompts = json.loads((ROOT / 'runs/review_715121/panel/prompts.json').read_text(encoding='utf-8'))['prompts'][:4]
    controls = {(r['doc'], r['seed']): r for r in
                map(json.loads, (ROOT / 'runs/continuation_review_20261101/samples.jsonl').read_text(encoding='utf-8').splitlines())
                if r['arm'] == 'uniform'}
    mean = torch.tensor(cache.mean, device='cuda')
    offsets = np.fromfile(cache.root / 'unit_offsets.u64', dtype='<u8')
    blob = (cache.root / 'units.bin').read_bytes(); counts = np.fromfile(cache.root / 'train_counts.u64', dtype='<u8')
    seen = {blob[int(offsets[i]):int(offsets[i + 1])] for i in np.flatnonzero(counts)}
    protocol = {'checkpoint': str(path), 'checkpoint_sha256': sha256(path), 'cumulative_updates': 28989,
                'weights': 'ema', 'steps': 32, 'seed': 42, 'prefix_units': 32, 'output_units': 128,
                'denoise_block': 16, 'committed_each_round': 1, 'new_noise_each_round': True,
                'history': 'continuous latents', 'precision': 'fp16', 'codec_precision': 'fp32',
                'docs': [q['doc'] for q in prompts], 'control': 'cached 717047 full-block output',
                'control_calls': 256, 'first_unit_calls': 4096,
                'source_hashes': {p: sha256(ROOT / p) for p in ['char_diff/flow.py', 'scripts/sample.py', 'scripts/compare_first_unit_commit.py']}}
    write_json(folder / 'protocol.json', protocol)
    # Short check of the changed KV/commit path against independent full-history inference.
    prefix = cache.story(prompts[0]['doc'])[:32]; trace = CommitTrace(model, check=True)
    actual = generate(trace, prefix, 3, steps=2, seed=42, commit_first=True)
    direct = generate(model, prefix, 3, steps=2, seed=42, use_cache=False, commit_first=True)
    full = generate(model, prefix, 16, steps=2, seed=42)
    torch.testing.assert_close(actual, direct, atol=3e-4, rtol=3e-4)
    assert torch.equal(actual[0], full[0]) and trace.calls == 6 and trace.cache_lengths == [34, 35, 36]
    # The affected default branch retains its previous full 128-unit output.
    torch.cuda.synchronize(); tick = time.perf_counter()
    with torch.autocast('cuda', dtype=torch.float16): baseline = generate(model, prefix, 128, 32, 42)
    baseline_parts = codec.decode(baseline * cache.scale + mean)
    torch.cuda.synchronize(); baseline_seconds = time.perf_counter() - tick
    previous = controls[(prompts[0]['doc'], 42)]
    assert b''.join(baseline_parts).hex() == previous['generated_hex']
    assert [len(v) for v in baseline_parts] == previous['unit_lengths']
    checks = {'cached_vs_full_history_max_abs': float((actual-direct).abs().max()), 'first_latent_equal_full_block': True,
              'full_denoise_width': 16, 'cache_lengths': trace.cache_lengths, 'default_control_byte_exact': True,
              'default_control_doc': prompts[0]['doc'], 'default_control_seconds': baseline_seconds}
    write_json(folder / 'targeted-checks.json', checks); print(json.dumps({'checks': checks}), flush=True)
    records = []; vectors = {}
    with (folder / 'pairs.jsonl').open('x', encoding='utf-8', buffering=1) as log:
        for q in prompts:
            control = controls[(q['doc'], 42)]; assert control['prompt'] == q['prompt']
            trace = CommitTrace(model, doc=q['doc']); torch.cuda.synchronize(); tick = time.perf_counter()
            with torch.autocast('cuda', dtype=torch.float16):
                z = generate(trace, cache.story(q['doc'])[:32], 128, 32, 42, commit_first=True)
            assert torch.isfinite(z).all() and trace.calls == 4096 and trace.committed == 128
            parts = codec.decode(z * cache.scale + mean); raw = b''.join(parts)
            torch.cuda.synchronize(); elapsed = time.perf_counter() - tick
            first = bytes.fromhex(control['generated_hex'])[:control['unit_lengths'][0]]
            assert parts[0] == first
            record = {'doc': q['doc'], 'seed': 42, 'prompt': q['prompt'], 'units': 128,
                      'first_unit_equal': True, 'control': control,
                      'first_only': {'generated': raw.decode('utf-8', errors='replace'), 'generated_hex': raw.hex(),
                                     'unit_hex': [v.hex() for v in parts], 'unit_lengths': [len(v) for v in parts],
                                     'utf8_valid': raw.decode('utf-8', errors='replace').encode('utf-8') == raw,
                                     'seen_training_units': sum(v in seen for v in parts),
                                     'empty_units': sum(not v for v in parts), 'seconds': elapsed}}
            records.append(record); log.write(json.dumps(record, ensure_ascii=False) + '\n')
            vectors[f"doc_{q['doc']}_seed42"] = z.cpu().numpy()
            np.savez(folder / 'latents.npz', **vectors)
            print(json.dumps({'completed_doc': q['doc'], 'seconds': elapsed, 'text': record['first_only']['generated']}, ensure_ascii=True), flush=True)
    summary = {'protocol': protocol, 'checks': checks, 'cases': len(records), 'groups': {}}
    for key in ['control', 'first_only']:
        summary['groups'][key] = {'units': 128 * len(records),
                                 'seen_training_units': sum(r[key]['seen_training_units'] for r in records),
                                 'utf8_valid': sum(r[key]['utf8_valid'] for r in records),
                                 'empty_units': sum(r[key]['empty_units'] for r in records)}
    summary['first_only_mean_seconds'] = float(np.mean([r['first_only']['seconds'] for r in records]))
    summary['first_doc_time_ratio'] = records[0]['first_only']['seconds'] / baseline_seconds
    write_json(folder / 'summary.json', summary)
    docs = ROOT / 'docs/experiments/first-unit-commit'; (docs / 'evidence').mkdir(parents=True, exist_ok=True)
    write_json(docs / 'evidence/summary.json', summary)
    lines = ['# 整块提交与只提交首unit：全部配对正文', '']
    for r in records:
        lines += [f"## doc{r['doc']} / seed42", '', '**前缀**', '', r['prompt'], '',
                  '**整块16unit提交（缓存控制）**', '', r['control']['generated'], '',
                  '**整块去噪，只提交首unit**', '', r['first_only']['generated'], '']
    (docs / 'samples.md').write_text('\n'.join(lines), encoding='utf-8')
    print(json.dumps({'summary': summary}), flush=True)


if __name__ == '__main__':
    main()
