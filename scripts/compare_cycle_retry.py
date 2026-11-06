"""Approved B1 R>0.1 single-retry experiment: common dev EM and four continuations."""
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
from char_diff.selection import CycleScorer, CycleRetry, needs_retry, prefer_retry
from char_diff.utils import ROOT, sha256, write_json
from compare_byte_em import diffusion_batch, diffusion_predict
from review_block_size import metrics, SampleProgress

OUT = ROOT / 'runs/cycle_retry_20261102'
DOCS = ROOT / 'docs/experiments/decoder-friendliness'
CONTROL = ROOT / 'runs/block_size_review_20261102'
CHECKPOINT = ROOT / 'runs/tinystories_x0_block1/last.pt'
THRESHOLD = .1


def read(path): return json.loads(path.read_text(encoding='utf-8'))
def lines(path): return [json.loads(s) for s in path.read_text(encoding='utf-8').splitlines()]


@torch.inference_mode()
def teacher(model, cache, codec, state):
    data = read(ROOT / 'runs/cross_model_byte_em_20261102/anchors.json'); cases = data['cases']
    controls = {(r['id'], r['seed']): r for r in lines(CONTROL / 'b1-teacher.jsonl')}
    scorer = CycleScorer(codec, state['normalizer'])
    streams = {s: torch.Generator(device='cuda').manual_seed(s) for s in (42, 123)}
    retries = {s: torch.Generator(device='cuda').manual_seed(s+1000003) for s in (42, 123)}
    rows = []; tick = time.perf_counter()
    with (OUT / 'teacher.jsonl').open('x', encoding='utf-8', buffering=1) as handle:
        for begin in range(0, len(cases), 64):
            group = [dict(c, block_length=1) for c in cases[begin:begin+64]]
            packed = diffusion_batch(cache, group)
            for seed in (42, 123):
                noise = torch.stack([torch.randn(16, 48, device='cuda', generator=streams[seed])[:1] for _ in group])
                z = diffusion_predict(model, packed, noise, 'euler')
                original = scorer(z[:, 0])
                flagged = [i for i, r in enumerate(original) if needs_retry(r, THRESHOLD)]
                alternatives = {}
                if flagged:
                    subpack = tuple(t[flagged] for t in packed)
                    noise2 = torch.stack([torch.randn(1, 48, device='cuda', generator=retries[seed]) for _ in flagged])
                    z2 = diffusion_predict(model, subpack, noise2, 'euler')
                    alternatives = dict(zip(flagged, scorer(z2[:, 0])))
                for i, case in enumerate(group):
                    old = controls[(case['id'], seed)]
                    assert original[i]['unit_hex'] == old['predicted_unit_hex'], 'Initial candidate must reproduce cached B1'
                    second = alternatives.get(i)
                    accepted = second is not None and prefer_retry(original[i], second)
                    selected = second if accepted else original[i]
                    row = {'id': case['id'], 'window': case['window'], 'seed': seed,
                           'target_bytes': old['target_bytes'], 'target_hex': case['target_hex'],
                           'exact': int(selected['unit_hex'] == case['target_hex']),
                           'control_exact': old['exact'], 'predicted_unit_hex': selected['unit_hex'],
                           'triggered': second is not None, 'accepted_retry': accepted,
                           'changed_bytes': selected['unit_hex'] != original[i]['unit_hex'],
                           'original': original[i], 'retry': second, 'selected': selected}
                    rows.append(row); handle.write(json.dumps(row) + '\n')
            if begin % 1024 == 0:
                print(json.dumps({'teacher_anchors_done': begin + len(group), 'total': len(cases)}), flush=True)
    assert len(rows) == 3690
    control_metrics = metrics(list(controls.values())); new_metrics = metrics(rows)
    windows = data['windows']; indices = {w: i for i, w in enumerate(windows)}
    counts = np.zeros((len(windows), 6))
    for r in rows:
        counts[indices[r['window']]] += [r['exact'], r['control_exact'], 1,
                                         r['exact']*r['target_bytes'], r['control_exact']*r['target_bytes'], r['target_bytes']]
    draws = np.random.default_rng(20261102).integers(0, len(windows), (5000, len(windows)))
    totals = counts[draws].sum(1)
    delta = 100 * np.stack([(totals[:, 0]-totals[:, 1])/totals[:, 2],
                            (totals[:, 3]-totals[:, 4])/totals[:, 5]], 1)
    report = {'control': control_metrics, 'cycle_retry': new_metrics,
              'initial_candidate_byte_exact': len(rows), 'retries': sum(r['triggered'] for r in rows),
              'accepted_retries': sum(r['accepted_retry'] for r in rows),
              'changed_units': sum(r['changed_bytes'] for r in rows),
              'incorrect_to_correct': sum(r['exact'] and not r['control_exact'] for r in rows),
              'correct_to_incorrect': sum(r['control_exact'] and not r['exact'] for r in rows),
              'invalid_selected': sum(not r['selected']['valid'] for r in rows),
              'mean_target_bytes': float(np.mean([r['target_bytes'] for r in rows])),
              'mean_generated_bytes_control': float(np.mean([r['original']['bytes'] for r in rows])),
              'mean_generated_bytes_retry': float(np.mean([r['selected']['bytes'] for r in rows])),
              'paired_window_bootstrap': {'draws': 5000, 'seed': 20261102,
                  'delta_pp': {key: {'observed': 100*(new_metrics[key]-control_metrics[key]),
                                     'ci95': np.quantile(delta[:, i], [.025, .975]).tolist()}
                               for i, key in enumerate(('unit_em', 'byte_weighted_unit_em'))}},
              'seconds': time.perf_counter() - tick}
    report['retry_fraction'] = report['retries']/len(rows)
    report['base_denoiser_unit_calls'] = 32*len(rows)
    report['additional_denoiser_unit_calls'] = 32*report['retries']
    write_json(OUT / 'teacher-summary.json', report)
    write_json(DOCS / 'evidence/retry-teacher.json', report)
    print(json.dumps({'teacher_summary': {k:v for k,v in report.items() if k not in ('control','cycle_retry')},
                      'em': {'control': control_metrics['unit_em'], 'retry': new_metrics['unit_em']}}), flush=True)


@torch.inference_mode()
def samples(model, cache, codec, state):
    panel = read(ROOT / 'runs/review_715121/panel/prompts.json')['prompts'][:4]
    controls = {r['doc']: r for r in lines(CONTROL / 'samples.jsonl') if r['arm']=='b1'}
    offsets = np.fromfile(cache.root / 'unit_offsets.u64', dtype='<u8')
    counts = np.fromfile(cache.root / 'train_counts.u64', dtype='<u8'); blob = (cache.root / 'units.bin').read_bytes()
    seen = {blob[int(offsets[i]):int(offsets[i+1])] for i in np.flatnonzero(counts)}
    mean = torch.tensor(cache.mean, device='cuda'); vectors = {}
    with (OUT / 'samples.jsonl').open('x', encoding='utf-8', buffering=1) as handle:
        for q in panel:
            assert q['prompt'] == controls[q['doc']]['prompt']
            selector = CycleRetry(CycleScorer(codec, state['normalizer']), THRESHOLD)
            traced = SampleProgress(model, 'cycle_retry', q['doc'])
            torch.cuda.synchronize(); tick = time.perf_counter()
            with torch.autocast('cuda', dtype=torch.float16):
                z = generate(traced, cache.story(q['doc'])[:32], 128, 32, 42, unit_selector=selector)
            assert torch.isfinite(z).all()
            units = codec.decode(z*cache.scale+mean); raw = b''.join(units)
            assert [u.hex() for u in units] == [r['selected']['unit_hex'] for r in selector.records]
            torch.cuda.synchronize(); elapsed = time.perf_counter()-tick
            stats = selector.summary()
            assert traced.calls == 32*(128+stats['retries'])
            # Until the first accepted retry, the paired sample must follow the saved trajectory exactly.
            control_units = controls[q['doc']]['unit_hex']
            first_change = next((r['unit'] for r in selector.records if r['accepted_retry']), 128)
            assert [u.hex() for u in units[:first_change]] == control_units[:first_change]
            record = {'doc': q['doc'], 'seed': 42, 'prompt': q['prompt'], 'units': 128,
                      'generated': raw.decode('utf-8', errors='replace'), 'generated_hex': raw.hex(),
                      'unit_hex': [u.hex() for u in units], 'bytes': len(raw),
                      'seen_training_units': sum(u in seen for u in units),
                      'utf8_valid': raw.decode('utf-8', errors='replace').encode('utf-8') == raw,
                      'empty_units': sum(not u for u in units), 'seconds': elapsed,
                      'denoiser_calls': traced.calls, 'selection': stats,
                      'unchanged_prefix_until_first_accepted_retry': first_change}
            handle.write(json.dumps(record, ensure_ascii=False)+'\n')
            write_json(OUT / f'trace-doc{q["doc"]}.json', selector.records)
            vectors[f'doc{q["doc"]}_seed42'] = z.cpu().numpy()
            np.savez(OUT / 'sample-latents.npz', **vectors)
            print(json.dumps({'sample_done': q['doc'], 'seconds': elapsed, 'selection': stats,
                              'text': record['generated']}, ensure_ascii=True), flush=True)
    summarize_samples()


def summarize_samples():
    rows = lines(OUT / 'samples.jsonl'); assert len(rows)==4
    controls = {r['doc']: r for r in lines(CONTROL / 'samples.jsonl') if r['arm']=='b1'}
    report = {'groups': {}}
    for label, group in [('control',[controls[r['doc']] for r in rows]),('cycle_retry',rows)]:
        report['groups'][label] = {'cases': len(group), 'units': sum(r['units'] for r in group),
                                  'seen_training_units': sum(r['seen_training_units'] for r in group),
                                  'utf8_valid': sum(r['utf8_valid'] for r in group),
                                  'empty_units': sum(r['empty_units'] for r in group),
                                  'mean_seconds': float(np.mean([r['seconds'] for r in group])),
                                  'total_bytes': sum(len(bytes.fromhex(r['generated_hex'])) for r in group),
                                  'denoiser_calls': sum(r['denoiser_calls'] for r in group)}
    report['selection'] = {key: sum(r['selection'][key] for r in rows)
                           for key in ['units','retries','accepted_retries','changed_units','invalid_selected']}
    report['retry_fraction'] = report['selection']['retries']/report['selection']['units']
    report['mean_time_ratio'] = report['groups']['cycle_retry']['mean_seconds']/report['groups']['control']['mean_seconds']
    write_json(OUT / 'sample-summary.json', report)
    write_json(DOCS / 'evidence/retry-samples.json', report)
    text = ['# B1闭环核验与单次重试：固定前缀配对续写', '',
            'seed42，前缀32units，续写128units，EMA、每次32步Euler、FP16模型/FP32 AE。R阈值0.1，每个位置最多重试1次，保留有效且R较小的候选。原B1输出复用缓存。', '']
    for r in rows:
        old = controls[r['doc']]
        text += [f'## doc {r["doc"]}', '', '**前缀**', '', r['prompt'], '',
                 '**B1原控制**', '', old['generated'], '', '**B1闭环核验与重试**', '', r['generated'], '',
                 f"重试{r['selection']['retries']}次，接受{r['selection']['accepted_retries']}次；已见unit {old['seen_training_units']}→{r['seen_training_units']}/128；耗时{old['seconds']:.2f}→{r['seconds']:.2f}秒。", '']
    (DOCS / 'retry-samples.md').write_text('\n'.join(text)+'\n',encoding='utf-8')
    print(json.dumps({'sample_summary':report}),flush=True)


@torch.inference_mode()
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase',choices=['all','teacher','samples','summarize'],default='all')
    args=parser.parse_args(); torch.set_num_threads(4); OUT.mkdir(parents=True,exist_ok=True)
    if args.phase=='summarize': summarize_samples(); return
    state=torch.load(CHECKPOINT,map_location='cpu',weights_only=False)
    cache=LatentCache(ROOT/'data/tinystories','dev',block=1); codec=Codec(device='cuda')
    assert state['cache_fingerprint']==cache.fingerprint and state['codec']==codec.fingerprint
    assert state['state']['step']==28989 and state['config']['model']['block']==1
    model=Denoiser(state['config']['model']).cuda().eval(); model.load_state_dict(state['ema'])
    protocol={'checkpoint':str(CHECKPOINT),'checkpoint_sha256':sha256(CHECKPOINT),'weights':'ema',
              'threshold':THRESHOLD,'max_retries':1,'steps':32,'sampler':'euler','block':1,
              'selection':'valid candidate with smaller standardized cycle RMSE; ties keep original',
              'both_invalid':'retain original and record invalid_selected',
              'retry_seed_offset':1000003,'primary_rng':'unaffected by retry draws',
              'teacher':{'anchors':1845,'windows':64,'seeds':[42,123],'batch':64,'precision':'fp32'},
              'samples':{'docs':[6501,6054,10846,4930],'seed':42,'prefix_units':32,'output_units':128,
                         'precision':'fp16 model, fp32 AE','history':'selected continuous latent / clean KV'},
              'source_hashes':{p:sha256(ROOT/p) for p in ['char_diff/flow.py','char_diff/selection.py',
                  'scripts/sample.py','scripts/compare_cycle_retry.py','tests/test_cycle_retry.py']}}
    write_json(OUT/'protocol.json',protocol)
    if args.phase in ('all','teacher'): teacher(model,cache,codec,state)
    if args.phase in ('all','samples'): samples(model,cache,codec,state)


if __name__=='__main__': main()
