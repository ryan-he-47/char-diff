"""Frozen-AE geometry and decoder-noise diagnostics on actual planned units.

No network fitting. All path families share a checkpoint-specific scalar scale
estimated from the same independent reference strings. Occurrence distributions
are retained; noise panels use stratified samples and explicit inclusion weights.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import gzip
import json
import math
from pathlib import Path
import random
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from scipy.spatial.distance import cdist
import torch

from char_diff.ae_quality import tensorize
from char_diff.utils import write_json
from char_diff.vendor.bltz.config import Cfg
from char_diff.vendor.bltz.models.autoencoder import ByteStringAE


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line]


def edit_distance(a, b):
    """Myers bit-vector Levenshtein distance; accepts bytes or Unicode strings."""
    if not a:
        return len(b)
    masks = defaultdict(int)
    for i, token in enumerate(a):
        masks[token] |= 1 << i
    pv, mv, score, high = ~0, 0, len(a), 1 << (len(a) - 1)
    for token in b:
        eq = masks[token]
        xv = eq | mv
        xh = (((eq & pv) + pv) ^ pv) | eq
        ph = mv | ~(xh | pv)
        mh = pv & xh
        score += bool(ph & high) - bool(mh & high)
        ph, mh = (ph << 1) | 1, mh << 1
        pv, mv = mh | ~(xv | ph), ph & xv
    return int(score)


def quantiles(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    return {str(p): float(np.quantile(x, p)) for p in [.05, .25, .5, .75, .95]} if len(x) else {}


def geometry(z, strings, lengths, *, max_queries=2048, seed=1234):
    """Global occurrence-weighted spectrum; local distances on distinct strings."""
    z = np.asarray(z, dtype=np.float64)
    centered = z - z.mean(0)
    cov = centered.T @ centered / len(z)
    eigenvalues = np.linalg.eigvalsh(cov).clip(0)
    total = float(eigenvalues.sum())
    spectrum = eigenvalues / max(total, 1e-30)
    positive = spectrum[spectrum > 1e-12]
    norms = np.sqrt(np.mean(centered ** 2, axis=1))
    scale = math.sqrt(total / z.shape[1])
    variance = np.diag(cov)
    kurtosis = np.mean(centered ** 4, axis=0) / np.maximum(variance ** 2, 1e-30) - 3
    result = {'occurrences': len(z), 'dimension': z.shape[1], 'own_scalar_scale': scale,
              'mean': z.mean(0).tolist(), 'eigenvalues': eigenvalues[::-1].tolist(),
              'participation_ratio': float(1 / np.sum(spectrum ** 2)) if total > 0 else 0.,
              'entropy_effective_rank': float(np.exp(-np.sum(positive * np.log(positive)))) if len(positive) else 0.,
              'positive_spectrum_condition': float(positive.max() / positive.min()) if len(positive) else None,
              'top8_energy': float(spectrum[-8:].sum()), 'top16_energy': float(spectrum[-16:].sum()),
              'coordinate_excess_kurtosis_mean': float(kurtosis.mean()),
              'centered_norm_rmse_quantiles': quantiles(norms),
              'byte_length_norm_correlation': float(np.corrcoef(lengths, norms)[0, 1])
                  if np.std(lengths) > 0 and norms.std() > 0 else None}
    unique = {}
    for i, text in enumerate(strings):
        unique.setdefault(text, i)
    unique_strings = list(unique)
    uz = z[list(unique.values())] / max(scale, 1e-30)
    count = len(uz)
    result['unique_strings'] = count
    if count < 4:
        result['local'] = {'status': 'too_few_unique_strings'}
        return result
    rng = np.random.default_rng(seed)
    query_ids = np.sort(rng.choice(count, size=min(max_queries, count), replace=False))
    k = min(16, count - 1)
    nearest, distances = [], []
    zero_pairs = 0
    for start in range(0, len(query_ids), 128):
        q = query_ids[start:start + 128]
        dist = cdist(uz[q], uz)
        dist[np.arange(len(q)), q] = np.inf
        zero_pairs += int((dist < 1e-10).sum())
        dist[dist < 1e-10] = np.inf
        inds = np.argpartition(dist, kth=k - 1, axis=1)[:, :k]
        values = np.take_along_axis(dist, inds, axis=1)
        order = np.argsort(values, axis=1)
        nearest.append(np.take_along_axis(inds, order, axis=1))
        distances.append(np.take_along_axis(values, order, axis=1))
    near = np.concatenate(nearest)
    radius = np.concatenate(distances)
    ratios = np.log(radius[:, 1] / radius[:, 0])
    valid_ratios = ratios[np.isfinite(ratios) & (ratios > 1e-10)]
    estimates = {}
    for kk in [6, 10, 16]:
        if kk <= k:
            den = np.log(radius[:, kk - 1:kk] / radius[:, :kk - 1]).sum(1)
            ids = (kk - 2) / np.maximum(den, 1e-30)
            estimates[str(kk)] = quantiles(ids[np.isfinite(den) & (den > 1e-10)])
    by_length = defaultdict(list)
    for idx, text in enumerate(unique_strings):
        by_length[len(text)].append(idx)
    nearest_edit, matched_random_edit = [], []
    for q, neighbor in zip(query_ids, near[:, 0]):
        a, b = unique_strings[q], unique_strings[neighbor]
        nearest_edit.append(edit_distance(a, b) / max(len(a), len(b)))
        pool = [idx for idx in by_length[len(a)] if idx != q]
        if pool:
            other = unique_strings[int(rng.choice(pool))]
            matched_random_edit.append(edit_distance(a, other) / max(len(a), len(other)))
    result['local'] = {'query_count': len(query_ids), 'neighbor_bank_unique': count,
                       'distinct_string_nearzero_pairs': zero_pairs,
                       'nearest_distance_quantiles': quantiles(radius[:, 0] / math.sqrt(z.shape[1])),
                       'twonn_log_ratio_mle': float(1 / valid_ratios.mean()) if len(valid_ratios) else None,
                       'knn_mle_dimension_quantiles': estimates,
                       'nearest_byte_edit_fraction_mean': float(np.mean(nearest_edit)),
                       'same_byte_length_random_edit_fraction_mean': float(np.mean(matched_random_edit))
                           if matched_random_edit else None,
                       'same_byte_length_random_pairs': len(matched_random_edit),
                       'distance_units': 'RMSE / own scalar scale; duplicate strings removed'}
    return result


def stratified_indices(rows, cap, seed=1234):
    strata = defaultdict(list)
    for i, row in enumerate(rows):
        strata[(row['language'], row['bytes'])].append(i)
    rng = random.Random(seed)
    ordered = {}
    for key, ids in sorted(strata.items()):
        ordered[key] = rng.sample(ids, len(ids))
    quota = dict.fromkeys(ordered, 0)
    left = min(cap, len(rows))
    while left:
        for key, ids in ordered.items():
            if quota[key] < len(ids):
                quota[key] += 1
                left -= 1
                if not left:
                    break
    indices, weights = [], []
    for key, ids in ordered.items():
        n = quota[key]
        for idx in ids[:n]:
            indices.append(idx)
            weights.append(len(ids) / n)
    return indices, weights


def pca_projection(z_all, z_sample, dimensions):
    """Occurrence-weighted PCA projection; full dimension is exact identity."""
    z_all = np.asarray(z_all, dtype=np.float64)
    mean = z_all.mean(0)
    centered = z_all - mean
    eigenvalues, axes = np.linalg.eigh(centered.T @ centered / len(z_all))
    principal = axes[:, ::-1]
    projections = {}
    for dim in dimensions:
        if dim < 1 or dim > z_all.shape[1]:
            raise ValueError('PCA retained dimensions outside latent width')
        if dim == z_all.shape[1]:
            projections[dim] = np.asarray(z_sample, dtype=np.float32).copy()
        else:
            basis = principal[:, :dim]
            projections[dim] = (mean + ((z_sample - mean) @ basis) @ basis.T).astype(np.float32)
    return projections


@torch.inference_mode()
def encode(ae, strings, device, batch_size):
    parts = []
    for start in range(0, len(strings), batch_size):
        ids, _, mask = tensorize(strings[start:start + batch_size], ae.l_max, device)
        parts.append(ae.encoder(ids, mask).float().cpu().numpy())
    return np.concatenate(parts)


def summarize_noise(rows):
    if not rows:
        return {'sample_count': 0}
    weights = np.array([r['weight'] for r in rows])
    lengths = np.array([r['bytes'] for r in rows])
    exact = np.array([r['exact'] for r in rows], dtype=float)
    valid_cycle = np.array([r['cycle_rmse'] is not None for r in rows])
    cycles = np.array([r['cycle_rmse'] or 0 for r in rows])
    return {'sample_count': len(rows), 'estimated_occurrence_weight': float(weights.sum()),
            'em_unit_weighted': float(weights @ exact / weights.sum()),
            'em_byte_weighted': float((weights * lengths) @ exact / (weights @ lengths)),
            'byte_edit_rate': float(sum(r['weight'] * r['byte_edit'] for r in rows) / (weights @ lengths)),
            'character_edit_rate_valid_utf8': float(sum(r['weight'] * r['char_edit'] for r in rows if r['char_edit'] is not None) /
               max(sum(r['weight'] * r['characters'] for r in rows if r['char_edit'] is not None), 1e-30)),
            'utf8_valid_fraction': float(sum(r['weight'] * r['utf8_valid'] for r in rows) / weights.sum()),
            'eos_present_fraction': float(sum(r['weight'] * r['eos_present'] for r in rows) / weights.sum()),
            'empty_fraction': float(sum(r['weight'] * r['empty'] for r in rows) / weights.sum()),
            'cycle_valid_fraction': float(weights @ valid_cycle / weights.sum()),
            'cycle_rmse_valid_mean': float((weights * valid_cycle) @ cycles / max(weights @ valid_cycle, 1e-30)),
            'actual_noise_rmse_mean': float(sum(r['weight'] * r['actual_noise_rmse'] for r in rows) / weights.sum())}


@torch.inference_mode()
def perturb(ae, z, rows, weights, scalar_scale, sigma, seed, device, batch_size):
    # A per-unit key couples perturbations across path variants and checkpoints.
    noisy, noise_rmse = [], []
    for vector, row in zip(z, rows):
        key = hashlib.sha256(f'{seed}:{row["hex"]}'.encode()).digest()
        rng = np.random.default_rng(int.from_bytes(key[:8], 'little'))
        epsilon = rng.standard_normal(z.shape[1]).astype(np.float32)
        noisy.append(vector + scalar_scale * sigma * epsilon)
        noise_rmse.append(float(sigma * np.sqrt(np.mean(epsilon ** 2))))
    noisy = np.asarray(noisy, dtype=np.float32)
    output = []
    for start in range(0, len(rows), batch_size):
        part = rows[start:start + batch_size]
        zz = torch.tensor(noisy[start:start + len(part)], device=device)
        k = ae.l_max + 1
        logits = ae.decoder(zz.repeat_interleave(k, 0), torch.arange(k, device=device).repeat(len(part))).view(len(part), k, 257)
        if not torch.isfinite(logits).all():
            raise FloatingPointError('Non-finite decoder logits')
        predictions = logits.argmax(-1).cpu().tolist()
        cycle_strings, cycle_ids = [], []
        chunk = []
        for i, (sample, prediction) in enumerate(zip(part, predictions)):
            target = bytes.fromhex(sample['hex'])
            eos = 256 in prediction
            decoded = bytes(prediction[:prediction.index(256)] if eos else prediction)
            try:
                text = decoded.decode('utf-8', errors='strict')
                utf8 = True
            except UnicodeDecodeError:
                text, utf8 = None, False
            row = {**sample, 'weight': weights[start + i], 'sigma': sigma, 'seed': seed,
                   'exact': prediction[:len(target) + 1] == list(target) + [256],
                   'decoded_hex': decoded.hex(), 'byte_edit': edit_distance(target, decoded),
                   'char_edit': edit_distance(target.decode('utf-8'), text) if utf8 else None,
                   'utf8_valid': utf8, 'eos_present': eos, 'empty': not decoded,
                   'cycle_rmse': None, 'actual_noise_rmse': noise_rmse[start + i]}
            if utf8 and eos and 1 <= len(decoded) <= ae.l_max:
                cycle_strings.append(decoded)
                cycle_ids.append(i)
            chunk.append(row)
        if cycle_strings:
            back = encode(ae, cycle_strings, device, batch_size)
            errors = np.sqrt(np.mean(((back - noisy[start + np.asarray(cycle_ids)]) / scalar_scale) ** 2, axis=1))
            for idx, error in zip(cycle_ids, errors):
                chunk[idx]['cycle_rmse'] = float(error)
        output.extend(chunk)
    return output


def grouped_noise(rows):
    groups = {'by_language': defaultdict(list), 'by_byte_length': defaultdict(list),
              'by_language_and_length': defaultdict(list)}
    for row in rows:
        groups['by_language'][row['language']].append(row)
        groups['by_byte_length'][str(row['bytes'])].append(row)
        groups['by_language_and_length'][f'{row["language"]}:{row["bytes"]}'].append(row)
    return {'overall': summarize_noise(rows), **{name: {key: summarize_noise(value) for key, value in sorted(group.items())}
            for name, group in groups.items()}}


def load_groups(paths, fold, variants):
    groups = defaultdict(list)
    source = []
    for path in paths:
        for record in read_jsonl(path):
            if record.get('fold', 'analysis') != fold or not record.get('units'):
                continue
            variant = record['variant']
            if variants and variant not in variants:
                continue
            key = (record['checkpoint_tag'], variant)
            source.append(record)
            for position, unit in enumerate(record['units']):
                hx = unit.get('hex', unit.get('unit_hex'))
                raw = bytes.fromhex(hx)
                groups[key].append({'doc_id': record['doc_id'], 'language': record['language'],
                                    'start': unit['start'], 'end': unit['end'], 'position': position,
                                    'hex': hx, 'bytes': len(raw), 'characters': len(raw.decode('utf-8'))})
    return groups, source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--paths', type=Path, nargs='+', required=True)
    parser.add_argument('--checkpoint', action='append', required=True, help='tag=path')
    parser.add_argument('--reference-units', type=Path, required=True)
    parser.add_argument('--fold', default='analysis')
    parser.add_argument('--variants', nargs='*')
    parser.add_argument('--cross', action='append', default=[], help='path_source:model_tag')
    parser.add_argument('--max-noise-units', type=int, default=2048)
    parser.add_argument('--max-neighbor-queries', type=int, default=2048)
    parser.add_argument('--sigmas', nargs='+', type=float, default=[0, .025, .05, .1, .2])
    parser.add_argument('--noise-seeds', nargs='+', type=int, default=[42, 123])
    parser.add_argument('--pca-dims', nargs='+', type=int, default=[8, 16, 24, 32, 40, 48])
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    started = time.perf_counter()
    checkpoints = dict(value.split('=', 1) for value in args.checkpoint)
    cross = [tuple(value.split(':', 1)) for value in args.cross]
    groups, _ = load_groups(args.paths, args.fold, args.variants)
    reference_rows = read_jsonl(args.reference_units)
    reference = [bytes.fromhex(r.get('hex', r.get('unit_hex'))) for r in reference_rows]
    if not groups or not reference:
        raise ValueError('Need nonempty path groups and independent reference strings')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result = {'protocol': {'fold': args.fold, 'reference_units_sha256': digest(args.reference_units),
                          'paths_sha256': {str(p): digest(p) for p in args.paths},
                          'script_sha256': digest(__file__), 'sigmas': args.sigmas, 'noise_seeds': args.noise_seeds,
                          'noise': 'iid standard Gaussian keyed by seed+unit bytes, z_noisy=z+s_ref*sigma*epsilon',
                          'reference': 'same independent strings encoded per checkpoint; all variants share mean and scalar scale',
                          'noise_sampling': 'language x exact byte length strata, bounded without replacement; N_cell/n_cell weights',
                          'decoder': 'FP32 argmax; exact target bytes plus true EOS slot; free readout stops at first EOS',
                          'geometry': 'all path occurrences for moments; distinct strings for local kNN',
                          'pca_projection': 'own occurrence-weighted center/covariance; keep leading eigenspace; full width is identity',
                          'pca_dims': args.pca_dims,
                          'max_noise_units': args.max_noise_units, 'cross': cross},
              'models': {}, 'experiments': {}}
    for model_tag, checkpoint in checkpoints.items():
        selected = {key: rows for key, rows in groups.items() if key[0] == model_tag or (key[0], model_tag) in cross}
        if not selected:
            continue
        saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
        ae = ByteStringAE(Cfg(saved['cfg'])).to(device=args.device, dtype=torch.float32).eval()
        ae.load_state_dict(saved['model'], strict=True)
        ae.requires_grad_(False)
        all_strings = list(dict.fromkeys(reference + [bytes.fromhex(row['hex']) for rows in selected.values() for row in rows]))
        vectors = encode(ae, all_strings, args.device, args.batch_size)
        lookup = {text: i for i, text in enumerate(all_strings)}
        rz = vectors[[lookup[text] for text in reference]].astype(np.float64)
        mean = rz.mean(0)
        scale = float(np.sqrt(np.mean((rz - mean) ** 2)))
        if not scale > 0:
            raise ValueError('Reference scalar scale must be positive')
        result['models'][model_tag] = {'checkpoint': str(checkpoint), 'checkpoint_sha256': digest(checkpoint),
                                       'step': saved.get('step'), 'reference_mean': mean.tolist(),
                                       'reference_scalar_scale': scale, 'reference_count': len(reference),
                                       'unique_encoded_strings': len(all_strings)}
        np.savez_compressed(args.output_dir / f'embeddings_{model_tag}.npz',
                            hex=np.array([text.hex() for text in all_strings]), z=vectors)
        for (source_tag, variant), rows in selected.items():
            name = f'{model_tag}__from_{source_tag}__{variant}'
            strings = [bytes.fromhex(row['hex']) for row in rows]
            z = vectors[[lookup[text] for text in strings]]
            g = geometry(z, strings, [r['bytes'] for r in rows], max_queries=args.max_neighbor_queries)
            normalized_eigenvalues = np.array(g['eigenvalues']) / scale ** 2
            g['linear_flow_mode_snr'] = {}
            for t in [.1, .25, .5, .75, .9]:
                snr = ((1 - t) / t) ** 2 * normalized_eigenvalues
                g['linear_flow_mode_snr'][str(t)] = {
                    'modes_snr_above_one': int((snr > 1).sum()),
                    'sum_snr_over_one_plus_snr': float((snr / (1 + snr)).sum())}
            ids, weights = stratified_indices(rows, args.max_noise_units)
            panel = [rows[i] for i in ids]
            panel_z = z[ids]
            experiment = {'model_tag': model_tag, 'path_source_tag': source_tag, 'variant': variant,
                          'documents': len({r['doc_id'] for r in rows}), 'units': len(rows),
                          'bytes': sum(r['bytes'] for r in rows),
                          'reference_scale': scale, 'own_to_reference_scale': g['own_scalar_scale'] / scale,
                          'geometry': g, 'noise_panel_units': len(panel), 'noise': {}}
            clean_records = None
            noise_file = args.output_dir / f'noise_{name}.jsonl.gz'
            with gzip.open(noise_file, 'wt', encoding='utf-8') as handle:
                for sigma in args.sigmas:
                    records = []
                    for seed in (args.noise_seeds if sigma else args.noise_seeds[:1]):
                        rr = perturb(ae, panel_z, panel, weights, scale, sigma, seed, args.device, args.batch_size)
                        records.extend(rr)
                        for row in rr:
                            handle.write(json.dumps(row, ensure_ascii=False, separators=(',', ':')) + '\n')
                    experiment['noise'][str(sigma)] = grouped_noise(records)
                    if sigma == 0:
                        clean_records = records
            projections = pca_projection(z, panel_z, args.pca_dims)
            experiment['pca_projection'] = {}
            with gzip.open(args.output_dir / f'pca_{name}.jsonl.gz', 'wt', encoding='utf-8') as handle:
                for dim, projected in projections.items():
                    if dim == z.shape[1] and clean_records is not None:
                        records = clean_records
                    else:
                        records = perturb(ae, projected, panel, weights, scale, 0., args.noise_seeds[0], args.device, args.batch_size)
                    experiment['pca_projection'][str(dim)] = grouped_noise(records)
                    for row in records:
                        handle.write(json.dumps({**row, 'pca_dims': dim}, ensure_ascii=False, separators=(',', ':')) + '\n')
            experiment['pca_full_width_matches_clean'] = (
                experiment['pca_projection'].get(str(z.shape[1]), {}).get('overall') ==
                grouped_noise(clean_records)['overall']) if clean_records is not None else None
            result['experiments'][name] = experiment
            write_json(args.output_dir / 'results.json', result)
            print(json.dumps({'event': 'group_complete', 'name': name, 'units': len(rows),
                              'geometry_pr': g['participation_ratio'],
                              'noise': {s: x['overall']['em_unit_weighted'] for s, x in experiment['noise'].items()},
                              'elapsed_seconds': time.perf_counter() - started}), flush=True)
        del ae
        if args.device.startswith('cuda'):
            torch.cuda.empty_cache()
    result['elapsed_seconds'] = time.perf_counter() - started
    write_json(args.output_dir / 'results.json', result)


if __name__ == '__main__':
    main()
