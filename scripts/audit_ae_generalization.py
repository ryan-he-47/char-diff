"""Balanced train/calibration/test reconstruction gap on frozen AE checkpoints."""
from __future__ import annotations
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import random
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from char_diff.ae_quality import AEQualityScorer
from char_diff.utils import write_json, sha256
from char_diff.vendor.bltz.config import Cfg
from char_diff.vendor.bltz.models.autoencoder import ByteStringAE
from scripts.evaluate_ae_lengths import span_index, aggregate


def build_panel(path, docs_per_language=64, per_cell=16):
    by_lang, ids = defaultdict(list), set()
    raw_bytes = 0
    with open(path, encoding='utf-8') as handle:
        for line in handle:
            row = json.loads(line)
            assert hashlib.sha256(row['text'].encode()).hexdigest() == row['id']
            ids.add(row['id']); raw_bytes += len(row['text'].encode())
            by_lang[row['language']].append(row)
    selected, support = [], defaultdict(list)
    for lang, rows in sorted(by_lang.items()):
        ranked = sorted(rows, key=lambda r: hashlib.sha256(('ae-gap-doc-v1:' + r['id']).encode()).digest())
        for row in ranked[:docs_per_language]:
            raw = row['text'].encode()
            _, starts = span_index(row['text'])
            i = len(selected); selected.append((row, raw, starts))
            for n in range(1, 33):
                if starts[n]: support[(lang, n)].append(i)
    rng = random.Random(20261105)
    panel = []
    for (lang, n), candidates in sorted(support.items()):
        for _ in range(per_cell):
            i = rng.choice(candidates); row, raw, starts = selected[i]
            start = rng.choice(starts[n]); text = raw[start:start+n].decode('utf-8')
            panel.append({'doc_id': row['id'], 'language': lang, 'bytes': n, 'text': text,
                          'characters': len(text), 'start_byte': start, 'end_byte': start+n})
    assert len(support) == len(by_lang) * 32, 'Missing matched language/length cell'
    return panel, ids, {'documents': len(ids), 'raw_bytes': raw_bytes, 'panel_documents': len(selected),
                        'language_counts': {k: len(v) for k,v in by_lang.items()}}


def summaries(rows):
    buckets, languages = defaultdict(list), defaultdict(list)
    for r in rows:
        lo = 1 + ((r['bytes']-1)//8)*8
        buckets[f'{lo}-{lo+7}'].append(r); languages[r['language']].append(r)
    return {'overall': aggregate(rows), 'length_buckets': {k: aggregate(v) for k,v in buckets.items()},
            'languages': {k: aggregate(v) for k,v in languages.items()}}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--train', type=Path, required=True); p.add_argument('--calibration', type=Path, required=True)
    p.add_argument('--test', type=Path, required=True); p.add_argument('--checkpoint', action='append', required=True)
    p.add_argument('--out', type=Path, required=True); p.add_argument('--device', default='cuda')
    a = p.parse_args(); torch.set_num_threads(4); a.out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter(); panels, id_sets, sources = {}, {}, {}
    for name in ['train','calibration','test']:
        source = getattr(a, name); panel, ids, meta = build_panel(source)
        panels[name], id_sets[name] = panel, ids
        sources[name] = {**meta, 'sha256': sha256(source), 'path': str(source)}
        write_json(a.out/f'panel_{name}.json', panel)
    overlaps = {f'{a}:{b}': len(id_sets[a]&id_sets[b]) for a,b in [('train','calibration'),('train','test'),('calibration','test')]}
    assert not any(overlaps.values()), 'Exact document leakage'
    result = {'protocol': {'sources': sources, 'document_overlaps': overlaps,
               'panel': '64 hash-ranked complete documents/language, 16 windows per language x exact length 1..32B; no AE selection',
               'scope': 'natural windows, balanced exact lengths; frozen FP32 byte+EOS evaluation',
               'script_sha256': sha256(__file__), 'test_used_for': 'generalization audit only, no parameter or lambda tuning'}, 'models': {}}
    for item in a.checkpoint:
        tag, file = item.split('=',1); state = torch.load(file, map_location='cpu', weights_only=False)
        ae = ByteStringAE(Cfg(state['cfg'])).to(a.device).eval(); ae.load_state_dict(state['model']); ae.requires_grad_(False)
        scorer = AEQualityScorer(ae, device=a.device, batch_size=512)
        outputs = {}
        for name, panel in panels.items():
            scores = scorer.score_records([r['text'].encode() for r in panel])
            rows = [{**r, **s, 'exact':s['feasible']} for r,s in zip(panel,scores)]
            write_json(a.out/f'rows_{tag}_{name}.json',rows)
            outputs[name] = summaries(rows)
        result['models'][tag] = {'checkpoint':file, 'sha256':sha256(file), 'step':state['step'], 'splits':outputs}
        write_json(a.out/'results.json',result)
        print(json.dumps({'model':tag,'overall_em':{k:v['overall']['em'] for k,v in outputs.items()}}),flush=True)
        del ae, scorer, state
    result['elapsed_seconds'] = time.perf_counter()-started; write_json(a.out/'results.json',result)


if __name__ == '__main__': main()
