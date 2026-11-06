"""Read-only decoder diagnostics on saved B1/B4 generation latents."""
import json
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from char_diff.codec import Codec
from char_diff.data import LatentCache
from char_diff.utils import ROOT, sha256, write_json


@torch.inference_mode()
def scores(codec, vectors):
    result = []
    k = codec.ae.l_max + 1
    for begin in range(0, len(vectors), 256):
        z = vectors[begin:begin + 256]
        logits = codec.ae.decoder(z.repeat_interleave(k, 0), torch.arange(k, device=z.device).repeat(len(z)))
        logp = logits.float().log_softmax(-1).reshape(len(z), k, 257)
        p = logp.exp(); h = -(p * logp).sum(-1) / np.log(2)
        top = p.topk(2, dim=-1).values
        best = logp.argmax(-1)
        for i in range(len(z)):
            ids = best[i].tolist()
            n = min(ids.index(256) if 256 in ids else k, codec.ae.l_max)
            raw = bytes(ids[:n])
            result.append({'hex': raw.hex(), 'bytes': n,
                           'mean_entropy_bits': float(h[i, :n].mean()) if n else None,
                           'max_entropy_bits': float(h[i, :n].max()) if n else None,
                           'min_byte_probability': float(top[i, :n, 0].min()) if n else None,
                           'min_probability_margin': float((top[i, :n, 0] - top[i, :n, 1]).min()) if n else None,
                           'mean_self_nll': float(-logp[i, :n + 1].max(-1).values.mean()),
                           'eos_probability': float(p[i, n, 256]), 'eos_argmax': ids[n] == 256})
    return result


def auc(values, positive):
    a = np.asarray(values)[positive]; b = np.asarray(values)[~positive]
    return float(((a[:, None] > b[None]) + .5 * (a[:, None] == b[None])).mean()) if len(a) and len(b) else None


@torch.inference_mode()
def main():
    torch.set_num_threads(4); tick = time.perf_counter()
    source = ROOT / 'runs/block_size_review_20261102'
    out = ROOT / 'runs/decoder_friendliness_probe_20261102'; out.mkdir(parents=True, exist_ok=True)
    cache = LatentCache(ROOT / 'data/tinystories', 'dev'); codec = Codec(device='cuda')
    samples = [json.loads(s) for s in (source / 'samples.jsonl').read_text(encoding='utf-8').splitlines()]
    archive = np.load(source / 'sample-latents.npz')
    rows = []; latent = []
    for sample in samples:
        name = f'{sample["arm"]}_doc{sample["doc"]}_seed42'
        latent.append(archive[name])
        rows.extend({'arm': sample['arm'], 'doc': sample['doc'], 'unit': i, 'hex': h}
                    for i, h in enumerate(sample['unit_hex']))
    x = torch.tensor(np.concatenate(latent), device='cuda')
    mean = torch.tensor(cache.mean, device='cuda'); z = x * cache.scale + mean
    raw = [bytes.fromhex(r['hex']) for r in rows]
    assert all(0 < len(v) <= codec.ae.l_max for v in raw)
    direct = scores(codec, z)
    assert [r['hex'] for r in direct] == [r['hex'] for r in rows]
    encoded = torch.tensor(codec.encode(raw), device='cuda')
    reencoded = scores(codec, encoded)
    cycle = ((z - encoded) / cache.scale).square().mean(-1).sqrt().cpu().tolist()
    offsets = np.fromfile(cache.root / 'unit_offsets.u64', dtype='<u8')
    counts = np.fromfile(cache.root / 'train_counts.u64', dtype='<u8'); blob = (cache.root / 'units.bin').read_bytes()
    seen = {blob[int(offsets[i]):int(offsets[i + 1])] for i in np.flatnonzero(counts)}
    for i, row in enumerate(rows):
        row.update(direct[i]); row['text'] = raw[i].decode('utf-8', errors='replace')
        row['seen_in_training'] = raw[i] in seen
        row['cycle_rmse_standardized'] = cycle[i]
        row['reencoded_entropy_bits'] = reencoded[i]['mean_entropy_bits']
        row['roundtrip_exact'] = reencoded[i]['hex'] == row['hex']
    with (out / 'units.jsonl').open('x', encoding='utf-8') as handle:
        for row in rows: handle.write(json.dumps(row, ensure_ascii=False) + '\n')
    fields = ['mean_entropy_bits', 'max_entropy_bits', 'mean_self_nll', 'cycle_rmse_standardized', 'reencoded_entropy_bits']
    summary = {'protocol': {'source': str(source), 'units': len(rows), 'models': ['b1', 'b4'],
                           'new_diffusion_generations': 0, 'precision': 'fp32', 'codec': codec.fingerprint,
                           'label': 'training-unseen byte string, an automatic lexical proxy; inspect examples separately',
                           'threshold_reference': 'existing verifier mean byte entropy >0.2 bits',
                           'source_hashes': {p: sha256(source / p) for p in ['samples.jsonl', 'sample-latents.npz']},
                           'script_sha256': sha256(Path(__file__))}, 'groups': {}}
    for arm in ['all', 'b1', 'b4']:
        group = [r for r in rows if arm == 'all' or r['arm'] == arm]
        unseen = np.array([not r['seen_in_training'] for r in group])
        flagged = np.array([r['mean_entropy_bits'] > .2 for r in group])
        result = {'units': len(group), 'training_unseen': int(unseen.sum()),
                  'roundtrip_exact': sum(r['roundtrip_exact'] for r in group),
                  'eos_argmax': sum(r['eos_argmax'] for r in group),
                  'meanH_gt_0.2': {'flagged': int(flagged.sum()), 'unseen_flagged': int((flagged & unseen).sum()),
                                   'seen_flagged': int((flagged & ~unseen).sum())}, 'signals': {}}
        for key in fields:
            values = np.array([r[key] for r in group])
            result['signals'][key] = {'unseen_detection_auc': auc(values, unseen),
                                      'seen_median': float(np.median(values[~unseen])),
                                      'unseen_median': float(np.median(values[unseen]))}
        summary['groups'][arm] = result
    unknown = [r for r in rows if not r['seen_in_training']]
    summary['low_entropy_unseen_examples'] = sorted(unknown, key=lambda r: r['mean_entropy_bits'])[:12]
    summary['high_entropy_unseen_examples'] = sorted(unknown, key=lambda r: r['mean_entropy_bits'], reverse=True)[:8]
    summary['seconds'] = time.perf_counter() - tick
    write_json(out / 'summary.json', summary)
    print(json.dumps({'groups': summary['groups'], 'seconds': summary['seconds'],
                      'low_entropy_unseen': [{k:r[k] for k in ['text', 'mean_entropy_bits', 'min_byte_probability',
                                                               'cycle_rmse_standardized', 'reencoded_entropy_bits']}
                                            for r in summary['low_entropy_unseen_examples'][:6]]}, ensure_ascii=True), flush=True)


if __name__ == '__main__': main()
