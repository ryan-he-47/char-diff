"""Compare compact NLL scoring with existing complete-document reference paths."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from char_diff.ae_nll import AENLLScorer, tensorize_compact
from char_diff.ae_quality import AEQualityScorer, tensorize
from char_diff.ae_paths import candidate_spans, score_candidates, shortest_path
from char_diff.utils import write_json, sha256
from char_diff.vendor.bltz.config import Cfg
from char_diff.vendor.bltz.models.autoencoder import ByteStringAE


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--reference-paths', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--batch-size', type=int, default=512)
    p.add_argument('--device', default='cuda')
    a = p.parse_args(); a.out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    state = torch.load(a.checkpoint, map_location='cpu', weights_only=False)
    ae = ByteStringAE(Cfg(state['cfg'])); ae.load_state_dict(state['model'])
    ae.requires_grad_(False)
    scorer = AENLLScorer(ae, device=a.device, batch_size=a.batch_size)
    reference = AEQualityScorer(ae, device=a.device, batch_size=a.batch_size)
    rows = [json.loads(s) for s in a.reference_paths.read_text(encoding='utf-8').splitlines() if s]
    fixture = [b'a\x00b', '苹果'.encode(), 'é'.encode(), '🙂'.encode(), b'x'*32]
    t0 = tensorize(fixture, ae.l_max, a.device)
    t1 = tensorize_compact(fixture, ae.l_max, a.device)
    assert all(torch.equal(x, y) for x, y in zip(t0, t1))
    candidates = list(dict.fromkeys(bytes.fromhex(r['source_hex'])[s:e]
                       for r in rows[:2] for s,e in candidate_spans(bytes.fromhex(r['source_hex']))))
    selected = candidates[:a.batch_size] + candidates[-a.batch_size:]
    old = reference(selected); new = scorer(selected)
    flips = sum(x.feasible != y.feasible for x,y in zip(old,new))
    max_error = max(abs(x.quality-y.quality) for x,y in zip(old,new))
    assert flips == 0 and max_error == 0, (flips, max_error)
    records = []; started = time.perf_counter()
    for row in rows:
        tick = time.perf_counter(); raw = bytes.fromhex(row['source_hex'])
        scores = score_candidates(raw, scorer, batch_size=a.batch_size)
        path = shortest_path(raw, scores, quality_weight=10)
        same = list(path.boundaries) == row['boundaries']
        assert b''.join(path.units) == raw
        record = {'doc_id':row['doc_id'], 'language':row['language'], 'source_bytes':len(raw),
                  'units':len(path.units), 'unique_candidates':len(scores), 'same_path':same,
                  'cost_error':path.cost-row['cost'], 'seconds':time.perf_counter()-tick}
        records.append(record)
        if len(records) % 12 == 0: print(json.dumps({'done':len(records),'seconds':time.perf_counter()-started}),flush=True)
    seconds = time.perf_counter()-started
    result = {'checkpoint_sha256':sha256(a.checkpoint), 'reference_sha256':sha256(a.reference_paths),
              'batch_size':a.batch_size, 'device':str(scorer.device), 'precision':'fp32',
              'tensorization_exact':True, 'score_test_candidates':len(selected),
              'score_feasibility_flips':flips, 'score_max_abs_error':max_error,
              'same_paths':sum(r['same_path'] for r in records), 'documents':len(rows),
              'seconds':seconds, 'source_bytes':sum(r['source_bytes'] for r in records), 'rows':records}
    write_json(a.out/'results.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='rows'}),flush=True)
    assert result['same_paths'] == len(rows), 'Review changed paths before deployment'


if __name__ == '__main__': main()
