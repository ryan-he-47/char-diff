"""Native-block next-unit teacher forcing and paired autoregressive samples."""
import argparse
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
from char_diff.model import Denoiser
from char_diff.utils import ROOT, sha256, write_json
from compare_byte_em import diffusion_batch, diffusion_predict

FOLDER = ROOT / 'runs/block_size_review_20261102'
OLD = ROOT / 'runs/cross_model_byte_em_20261102'
DOCS = ROOT / 'docs/experiments/block-size'
ARMS = {f'b{b}': ROOT / f'runs/tinystories_x0_block{b}/last.pt' for b in (1, 4)}


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def lines(path):
    return [json.loads(s) for s in path.read_text(encoding='utf-8').splitlines()]


def metrics(rows):
    def count(values):
        n = len(values); total = sum(r['target_bytes'] for r in values)
        hits = sum(r['exact'] for r in values)
        hit_bytes = sum(r['exact'] * r['target_bytes'] for r in values)
        return {'cases': n, 'exact': hits, 'target_bytes': total, 'exact_bytes': hit_bytes,
                'unit_em': hits / n if n else None,
                'byte_weighted_unit_em': hit_bytes / total if total else None}
    return {**count(rows), 'by_seed': {str(s): count([r for r in rows if r['seed'] == s]) for s in (42, 123)},
            'by_length': {f'{lo}-{hi}': count([r for r in rows if lo <= r['target_bytes'] <= hi])
                          for lo, hi in ((1, 1), (2, 2), (3, 4), (5, 8), (9, 16), (17, 32))}}


def load_model(path, cache, codec):
    state = torch.load(path, map_location='cpu', weights_only=False)
    assert state['state']['step'] == 28989
    assert state['cache_fingerprint'] == cache.fingerprint and state['codec'] == codec.fingerprint
    assert state['config']['train'].get('prediction_type', 'x0') == 'x0'
    model = Denoiser(state['config']['model']).cuda().eval()
    model.load_state_dict(state['ema'])
    return model, state


def native_cases(cases, block):
    # Histories and scoring targets are fixed. Only the native noisy width changes.
    return [dict(c, block_length=min(block, c['block_length'])) for c in cases]


@torch.inference_mode()
def check_adapter(model, cache, cases):
    b = cache.block
    chosen = [min(cases, key=lambda c: len(c['history_ids'])),
              max(cases, key=lambda c: len(c['history_ids'])),
              next(c for c in cases if c['block_length'] < b)] if b > 1 else [
                  min(cases, key=lambda c: len(c['history_ids'])),
                  max(cases, key=lambda c: len(c['history_ids']))]
    chosen = native_cases(chosen, b)
    noise = torch.zeros(len(chosen), b, 48, device='cuda')
    for i, case in enumerate(chosen):
        rng = torch.Generator(device='cuda').manual_seed(700 + i)
        length = case['block_length']
        noise[i, :length] = torch.randn(length, 48, device='cuda', generator=rng)
    actual = diffusion_predict(model, diffusion_batch(cache, chosen), noise, 'euler', steps=2)
    errors = []
    for i, case in enumerate(chosen):
        length = case['block_length']
        expected = generate(model, cache.vectors[case['history_ids']], length, steps=2, seed=700 + i)
        torch.testing.assert_close(actual[i, :length], expected, atol=3e-4, rtol=3e-4)
        errors.append(float((actual[i, :length] - expected).abs().max()))
    return {'native_block': b, 'case_ids': [c['id'] for c in chosen],
            'block_lengths': [c['block_length'] for c in chosen], 'max_abs_vs_generate': errors}


@torch.inference_mode()
def teacher(batch):
    data = read(OLD / 'anchors.json'); cases = data['cases']
    codec = Codec(device='cuda')
    protocol = {'split': 'dev', 'windows': data['windows'], 'anchors': len(cases), 'seeds': [42, 123],
                'steps_per_native_block': 32, 'weights': 'ema', 'precision': 'fp32', 'batch': batch,
                'history': 'identical saved true history at each common B16 anchor, <=192 baseline BPE tokens',
                'score': 'entire decoded first native AE unit equals entire target byte string',
                'byte_weighting': 'sum(target_bytes * entire_unit_hit) / sum(target_bytes)',
                'noise': 'pure Gaussian; draw old 16x48 noise per anchor, retain first B positions',
                'length': 'min(native B, saved original block length); target bytes used only in scoring',
                'anchors_file': str(OLD / 'anchors.json'), 'anchors_sha256': sha256(OLD / 'anchors.json'),
                'control_file': str(OLD / 'x0_euler.jsonl'), 'control_sha256': sha256(OLD / 'x0_euler.jsonl')}
    write_json(FOLDER / 'teacher-protocol.json', protocol)
    reference = {c['id']: c for c in cases}
    all_rows = {'b16': []}; report = {'protocol': protocol, 'models': {}, 'checks': {}}
    # Reuse the already generated B16 first units, and apply this strict unit score.
    for r in lines(OLD / 'x0_euler.jsonl'):
        c = reference[r['id']]; hit = int(r['native_unit_hex'] == c['target_hex'])
        assert hit == r['native_unit_exact']
        all_rows['b16'].append({'id': r['id'], 'window': r['window'], 'seed': r['seed'],
                               'predicted_unit_hex': r['native_unit_hex'], 'target_hex': c['target_hex'],
                               'target_bytes': r['target_bytes'], 'exact': hit})
    report['models']['b16'] = {'cached': True, 'metrics': metrics(all_rows['b16'])}
    for arm, path in ARMS.items():
        b = int(arm[1:]); cache = LatentCache(ROOT / 'data/tinystories', 'dev', block=b)
        model, state = load_model(path, cache, codec)
        assert model.cfg['block'] == b
        report['checks'][arm] = check_adapter(model, cache, cases)
        write_json(FOLDER / 'targeted-checks.json', report['checks'])
        trajectory = lines(path.parent / 'train.jsonl'); updates = [r for r in trajectory if r['updated']]
        training = {'checkpoint': str(path), 'sha256': sha256(path), 'state': state['state'],
                    'source_step': state['lineage']['source_step'], 'successful_updates': len(updates),
                    'skips': len(trajectory) - len(updates),
                    'first200_loss': float(np.mean([r['loss'] for r in updates[:200]])),
                    'last200_loss': float(np.mean([r['loss'] for r in updates[-200:]])),
                    'result': read(path.parent / 'result.json'), 'dev_ema': read(path.parent / 'dev-ema.json')}
        assert len(updates) == 8376
        mean = torch.tensor(cache.mean, device='cuda')
        streams = {s: torch.Generator(device='cuda').manual_seed(s) for s in (42, 123)}
        rows = []; tick = time.perf_counter()
        with (FOLDER / f'{arm}-teacher.jsonl').open('x', encoding='utf-8', buffering=1) as handle:
            for begin in range(0, len(cases), batch):
                group = native_cases(cases[begin:begin + batch], b)
                packed = diffusion_batch(cache, group)
                for seed in (42, 123):
                    noise = torch.stack([torch.randn(16, 48, device='cuda', generator=streams[seed])[:b] for _ in group])
                    z = diffusion_predict(model, packed, noise, 'euler')
                    # Only the first unit is scored; all B noisy positions participated in denoising.
                    predicted = codec.decode(z[:, 0] * cache.scale + mean)
                    for c, raw in zip(group, predicted):
                        target = bytes.fromhex(c['target_hex'])
                        row = {'id': c['id'], 'window': c['window'], 'seed': seed,
                               'predicted_unit_hex': raw.hex(), 'target_hex': target.hex(),
                               'target_bytes': len(target), 'exact': int(raw == target)}
                        rows.append(row); handle.write(json.dumps(row) + '\n')
                if begin % (batch * 16) == 0:
                    print(json.dumps({'teacher': arm, 'anchors_done': begin + len(group)}), flush=True)
        all_rows[arm] = rows
        report['models'][arm] = {'training': training, 'metrics': metrics(rows), 'seconds': time.perf_counter() - tick}
        write_json(FOLDER / 'teacher-progress.json', report)
        print(json.dumps({'teacher_done': arm, 'metrics': report['models'][arm]['metrics']}), flush=True)
        del model, state; torch.cuda.empty_cache()
    expected = {(c['id'], s) for c in cases for s in (42, 123)}
    for rows in all_rows.values():
        assert len(rows) == len(expected) and {(r['id'], r['seed']) for r in rows} == expected
    # Paired window bootstrap keeps both seeds and neighbouring targets together.
    windows = data['windows']; indices = {w: i for i, w in enumerate(windows)}
    draws = np.random.default_rng(20261102).integers(0, len(windows), (5000, len(windows)))
    rates = {}
    for arm, rows in all_rows.items():
        counts = np.zeros((len(windows), 4))
        for r in rows:
            counts[indices[r['window']]] += [r['exact'], 1, r['exact'] * r['target_bytes'], r['target_bytes']]
        summed = counts[draws].sum(1)
        rates[arm] = summed[:, [0, 2]] / summed[:, [1, 3]]
    report['paired_window_bootstrap'] = {'draws': 5000, 'seed': 20261102, 'delta_vs_b16_pp': {}}
    for arm in ARMS:
        report['paired_window_bootstrap']['delta_vs_b16_pp'][arm] = {
            key: {'observed': 100 * (report['models'][arm]['metrics'][key] - report['models']['b16']['metrics'][key]),
                  'ci95': np.quantile(100 * (rates[arm][:, i] - rates['b16'][:, i]), [.025, .975]).tolist()}
            for i, key in enumerate(('unit_em', 'byte_weighted_unit_em'))}
    write_json(FOLDER / 'teacher-summary.json', report)
    write_json(DOCS / 'evidence/final-teacher.json', report)


class SampleProgress(torch.nn.Module):
    def __init__(self, model, arm, doc):
        super().__init__(); self.base = model; self.cfg = model.cfg
        self.arm = arm; self.doc = doc; self.units = 0; self.calls = 0

    def forward(self, x, *args, **kwargs):
        if not kwargs.get('return_cache', False): self.calls += 1
        output = self.base(x, *args, **kwargs)
        if kwargs.get('return_cache', False) and kwargs.get('past') is not None:
            self.units += x.shape[1]
            if self.units % 32 == 0:
                print(json.dumps({'sample': self.arm, 'doc': self.doc, 'units': self.units}), flush=True)
        return output


@torch.inference_mode()
def samples():
    cache = LatentCache(ROOT / 'data/tinystories', 'dev'); codec = Codec(device='cuda')
    mean = torch.tensor(cache.mean, device='cuda')
    panel = read(ROOT / 'runs/review_715121/panel/prompts.json')['prompts'][:4]
    controls = {r['doc']: r for r in lines(ROOT / 'runs/continuation_review_20261101/samples.jsonl')
                if r['arm'] == 'uniform' and r['seed'] == 42}
    offsets = np.fromfile(cache.root / 'unit_offsets.u64', dtype='<u8')
    blob = (cache.root / 'units.bin').read_bytes(); counts = np.fromfile(cache.root / 'train_counts.u64', dtype='<u8')
    seen = {blob[int(offsets[i]):int(offsets[i + 1])] for i in np.flatnonzero(counts)}
    protocol = {'docs': [q['doc'] for q in panel], 'seed': 42, 'prefix_units': 32, 'output_units': 128,
                'steps': 32, 'weights': 'ema', 'precision': 'fp16 denoiser, fp32 codec',
                'history': 'continuous generated latents, clean KV cache', 'commit': 'full native block',
                'control': 'cached B16 uniform717047, same prefixes/seed/precision',
                'checkpoint_sha256': {arm: sha256(path) for arm, path in ARMS.items()}}
    write_json(FOLDER / 'sample-protocol.json', protocol)
    records = []; vectors = {}
    with (FOLDER / 'samples.jsonl').open('x', encoding='utf-8', buffering=1) as handle:
        for arm, path in ARMS.items():
            model, state = load_model(path, cache, codec); b = model.cfg['block']
            for q in panel:
                assert q['prompt'] == controls[q['doc']]['prompt']
                traced = SampleProgress(model, arm, q['doc'])
                torch.cuda.synchronize(); tick = time.perf_counter()
                with torch.autocast('cuda', dtype=torch.float16):
                    z = generate(traced, cache.story(q['doc'])[:32], 128, 32, 42)
                assert torch.isfinite(z).all() and traced.calls == (128 // b) * 32
                parts = codec.decode(z * cache.scale + mean); raw = b''.join(parts)
                torch.cuda.synchronize(); elapsed = time.perf_counter() - tick
                row = {'arm': arm, 'block': b, 'doc': q['doc'], 'seed': 42, 'prompt': q['prompt'],
                       'units': 128, 'denoiser_calls': traced.calls, 'seconds': elapsed,
                       'generated': raw.decode('utf-8', errors='replace'), 'generated_hex': raw.hex(),
                       'unit_hex': [p.hex() for p in parts], 'seen_training_units': sum(p in seen for p in parts),
                       'empty_units': sum(not p for p in parts),
                       'utf8_valid': raw.decode('utf-8', errors='replace').encode('utf-8') == raw}
                handle.write(json.dumps(row, ensure_ascii=False) + '\n'); records.append(row)
                vectors[f'{arm}_doc{q["doc"]}_seed42'] = z.cpu().numpy()
                np.savez(FOLDER / 'sample-latents.npz', **vectors)
                print(json.dumps({'sample_done': arm, 'doc': q['doc'], 'seconds': elapsed,
                                  'text': row['generated']}, ensure_ascii=True), flush=True)
            del model, state; torch.cuda.empty_cache()
    summarize_samples()


def summarize_samples():
    """Finalize saved samples without repeating GPU inference."""
    panel = read(ROOT / 'runs/review_715121/panel/prompts.json')['prompts'][:4]
    controls = {r['doc']: r for r in lines(ROOT / 'runs/continuation_review_20261101/samples.jsonl')
                if r['arm'] == 'uniform' and r['seed'] == 42}
    records = lines(FOLDER / 'samples.jsonl'); protocol = read(FOLDER / 'sample-protocol.json')
    assert len(records) == 8 and {(r['arm'], r['doc']) for r in records} == {
        (arm, q['doc']) for arm in ARMS for q in panel}
    groups = {'b16': [controls[q['doc']] for q in panel]}
    groups.update({arm: [r for r in records if r['arm'] == arm] for arm in ARMS})
    summary = {'protocol': protocol, 'groups': {}}
    for arm, rows in groups.items():
        durations = [r['seconds'] for r in rows if r.get('seconds') is not None]
        summary['groups'][arm] = {'cases': len(rows), 'units': sum(r['units'] for r in rows),
                                 'seen_training_units': sum(r['seen_training_units'] for r in rows),
                                 'empty_units': sum(r['empty_units'] for r in rows),
                                 'utf8_valid': sum(r['utf8_valid'] for r in rows),
                                 'timed_cases': len(durations),
                                 'mean_seconds': float(np.mean(durations)) if durations else None}
    old_timing = read(ROOT / 'runs/first_unit_commit_20261102/targeted-checks.json')
    first_doc = panel[0]['doc']
    assert old_timing['default_control_doc'] == first_doc and old_timing['default_control_byte_exact']
    summary['first_doc_timing'] = {
        'doc': first_doc, 'control_seconds': old_timing['default_control_seconds'],
        'control_source': 'previous byte-exact B16 reproduction in first-unit-commit experiment',
        'arms': {arm: {'seconds': next(r['seconds'] for r in groups[arm] if r['doc'] == first_doc),
                       'ratio_vs_control': next(r['seconds'] for r in groups[arm] if r['doc'] == first_doc)
                       / old_timing['default_control_seconds']} for arm in ARMS}}
    write_json(FOLDER / 'sample-summary.json', summary)
    write_json(DOCS / 'evidence/final-samples.json', summary)
    text = ['# B=16 / B=4 / B=1：固定前缀自回归续写', '',
            '4个既有dev前缀，seed42，前缀32units，续写128units，EMA，每个原生块32步；连续latent历史，FP16模型/FP32字节读出。B16为缓存控制。', '']
    for q in panel:
        text += [f'## doc {q["doc"]}', '', '**前缀**', '', q['prompt'], '']
        for arm in ('b16', 'b4', 'b1'):
            r = next(r for r in groups[arm] if r['doc'] == q['doc'])
            text += [f'**{arm.upper()}**', '', r['generated'], '']
    (DOCS / 'samples.md').write_text('\n'.join(text) + '\n', encoding='utf-8')
    print(json.dumps({'sample_summary': summary}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=['teacher', 'samples', 'summarize', 'all'], default='all')
    parser.add_argument('--batch', type=int, default=64)
    args = parser.parse_args(); torch.set_num_threads(4)
    FOLDER.mkdir(parents=True, exist_ok=True)
    hash_file = 'report-source-hashes.json' if args.phase == 'summarize' else 'source-hashes.json'
    write_json(FOLDER / hash_file, {p: sha256(ROOT / p) for p in
               ['scripts/review_block_size.py', 'scripts/compare_byte_em.py', 'char_diff/flow.py', 'char_diff/model.py']})
    if args.phase in ('teacher', 'all'): teacher(args.batch)
    if args.phase in ('samples', 'all'): samples()
    if args.phase == 'summarize': summarize_samples()
