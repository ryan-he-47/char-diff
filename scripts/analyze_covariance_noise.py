"""Frozen AE decoder noise ablation, reusing the existing occurrence panels.

Fit covariance only on the shared reference strings already in the saved NPZ.
Compare covariance-aligned and Haar-rotated noise with the saved isotropic
readout. No model fitting, new sampling, whitening, or baseline inference.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import gzip
import hashlib
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch

from scripts.analyze_path_latents import (
    ByteStringAE, Cfg, digest, grouped_noise, perturb, read_jsonl, write_json,
)

SAMPLE_KEYS = ('doc_id', 'language', 'start', 'end', 'position', 'hex',
               'bytes', 'characters')
ENERGY_KEYS = ('actual_noise_mse', 'pc_tail8_noise_energy',
               'constructed_noise_mse', 'constructed_pc_tail8_noise_energy')


def haar_orthogonal(dimension, seed):
    rng = np.random.default_rng(seed)
    q, r = np.linalg.qr(rng.standard_normal((dimension, dimension)))
    signs = np.where(np.diag(r) < 0, -1., 1.)
    return q * signs[None, :]


def covariance_shape(vectors):
    z = np.asarray(vectors, dtype=np.float64)
    mean = z.mean(0)
    centered = z - mean
    covariance = centered.T @ centered / len(z)
    values, axes = np.linalg.eigh(covariance)
    if values[0] < -1e-10 * max(1., values[-1]):
        raise ValueError('Reference covariance has a materially negative eigenvalue')
    clipped = values.clip(min=0)
    trace = float(clipped.sum())
    if trace <= 0:
        raise ValueError('Reference covariance has no positive variance')
    dimension = z.shape[1]
    shape = (axes * np.sqrt(clipped * dimension / trace)[None, :]) @ axes.T
    return {'mean': mean, 'covariance': covariance, 'eigenvalues_raw': values,
            'eigenvalues': clipped, 'axes': axes, 'shape': shape,
            'trace': trace, 'scalar_scale': np.sqrt(trace / dimension)}


def mathematical_checks():
    """Exactly two focused checks: covariance energy and rotation identities."""
    dimension = 48
    q = haar_orthogonal(dimension, 20261104)
    values = np.linspace(.003, 2., dimension)
    covariance = (q * values[None, :]) @ q.T
    # +/- columns give exact population covariance, without Monte Carlo error.
    support = np.sqrt(dimension) * (q * np.sqrt(values)[None, :]).T
    fit = covariance_shape(np.concatenate((support, -support)))
    a = fit['shape']
    np.testing.assert_allclose(a @ a.T, dimension * covariance / values.sum(),
                               atol=2e-14, rtol=2e-13)
    np.testing.assert_allclose(np.sum(a * a) / dimension, 1., atol=2e-14)
    sigma = .05
    probes = np.sqrt(dimension) * np.eye(dimension)
    delta = sigma * probes @ a.T
    np.testing.assert_allclose(np.mean(delta ** 2), sigma ** 2, atol=1e-16)
    matrix = {'name': 'covariance_shape_and_expected_energy', 'passed': True,
              'shape_trace_over_dimension': float(np.sum(a * a) / dimension),
              'exact_probe_mse': float(np.mean(delta ** 2)), 'target_mse': sigma ** 2}

    rotated = delta @ q.T
    np.testing.assert_allclose(q.T @ q, np.eye(dimension), atol=2e-14)
    np.testing.assert_allclose(np.sum(rotated ** 2, axis=1),
                               np.sum(delta ** 2, axis=1), atol=2e-15, rtol=2e-13)
    np.testing.assert_allclose(np.linalg.eigvalsh(q @ covariance @ q.T),
                               values, atol=2e-14, rtol=2e-13)
    rotation = {'name': 'haar_rotation_pointwise_energy_and_spectrum', 'passed': True,
                'max_pointwise_squared_norm_error': float(np.max(np.abs(
                    np.sum(rotated ** 2, axis=1) - np.sum(delta ** 2, axis=1))))}
    return [matrix, rotation]


def epsilon_for(rows, dimension):
    output = []
    for row in rows:
        key = hashlib.sha256(f'{row["seed"]}:{row["hex"]}'.encode()).digest()
        rng = np.random.default_rng(int.from_bytes(key[:8], 'little'))
        # Preserve the original iid directions' FP32 conversion exactly.
        output.append(rng.standard_normal(dimension).astype(np.float32))
    return np.asarray(output)


def append_energies(records, constructed, realized, tail_axes):
    made_mse = np.mean(constructed ** 2, axis=1)
    actual_mse = np.mean(realized ** 2, axis=1)
    made_tail = np.sum((constructed @ tail_axes) ** 2, axis=1)
    actual_tail = np.sum((realized @ tail_axes) ** 2, axis=1)
    for row, mse, tail, intended_mse, intended_tail in zip(
            records, actual_mse, actual_tail, made_mse, made_tail):
        row['actual_noise_mse'] = float(mse)
        row['pc_tail8_noise_energy'] = float(tail)
        row['constructed_noise_mse'] = float(intended_mse)
        row['constructed_pc_tail8_noise_energy'] = float(intended_tail)


def energy_summary(rows):
    weights = np.asarray([r['weight'] for r in rows], dtype=np.float64)
    byte_weights = weights * np.asarray([r['bytes'] for r in rows])
    result = {}
    for key in ENERGY_KEYS:
        values = np.asarray([r[key] for r in rows])
        result[key + '_unit_weighted_mean'] = float(weights @ values / weights.sum())
        result[key + '_byte_weighted_mean'] = float(byte_weights @ values / byte_weights.sum())
    total = sum(r['weight'] * r['actual_noise_mse'] * 48 for r in rows)
    result['pc_tail8_fraction_of_actual_energy'] = (
        sum(r['weight'] * r['pc_tail8_noise_energy'] for r in rows) / total if total else None)
    return result


def grouped_with_energy(rows):
    result = grouped_noise(rows)
    result['overall'].update(energy_summary(rows))
    by_language, by_length, by_both = defaultdict(list), defaultdict(list), defaultdict(list)
    for row in rows:
        by_language[row['language']].append(row)
        by_length[str(row['bytes'])].append(row)
        by_both[f'{row["language"]}:{row["bytes"]}'].append(row)
    for name, groups in [('by_language', by_language), ('by_byte_length', by_length),
                         ('by_language_and_length', by_both)]:
        for key, members in groups.items():
            result[name][key].update(energy_summary(members))
    return result


def load_sources(source_dirs):
    sources, models, experiments = [], {}, {}
    for directory in source_dirs:
        path = directory / 'results.json'
        saved = json.loads(path.read_text(encoding='utf-8'))
        sources.append({'path': str(path), 'sha256': digest(path), 'content': saved})
        for tag, model in saved['models'].items():
            if tag in models and model['checkpoint_sha256'] != models[tag]['checkpoint_sha256']:
                raise ValueError('Source runs disagree on checkpoint identity')
            models.setdefault(tag, {**model, 'npz': directory / f'embeddings_{tag}.npz'})
        for name, experiment in saved['experiments'].items():
            if name in experiments:
                raise ValueError(f'Duplicate experiment {name}')
            experiments[name] = (directory, experiment)
    return sources, models, experiments


def write_rows(handle, rows):
    for row in rows:
        handle.write(json.dumps(row, ensure_ascii=False, separators=(',', ':')) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-dirs', type=Path, nargs='+')
    parser.add_argument('--reference-units', type=Path)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--sigmas', type=float, nargs='+', default=[.025, .05])
    parser.add_argument('--rotation-seed', type=int, default=20261104)
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.check_only:
        checks = mathematical_checks()
        write_json(args.output_dir / 'math_checks.json', {'checks': checks})
        print(json.dumps({'mathematical_checks': checks}), flush=True)
        return
    if not args.source_dirs or not args.reference_units:
        parser.error('--source-dirs and --reference-units are required for inference')
    torch.set_num_threads(4)
    started = time.perf_counter()
    sources, models, experiments = load_sources(args.source_dirs)
    reference_rows = read_jsonl(args.reference_units)
    reference_hex = [r.get('hex', r.get('unit_hex')) for r in reference_rows]
    reference_hash = digest(args.reference_units)
    for source in sources:
        if source['content']['protocol']['reference_units_sha256'] != reference_hash:
            raise ValueError('Reference file differs from the original protocol')
    result = {'protocol': {
        'source_runs': [{k: v for k, v in item.items() if k != 'content'} for item in sources],
        'reference_units_sha256': reference_hash, 'script_sha256': digest(__file__),
        'sigmas': args.sigmas, 'rotation_seed': args.rotation_seed,
        'panels': 'Exact original records, seed+hex iid directions, occurrence weights; no new sample selection',
        'covariance': 'Population covariance of the shared 3840 reference strings from saved FP32 embeddings',
        'shape': 'A=sqrt(d/tr(C_ref))*C_ref^(1/2); numerical negative eigenvalues clipped to zero only',
        'aligned': 'delta=sigma*s_ref*A*epsilon',
        'rotated': 'delta_rot=Q*delta_aligned; same Haar Q across checkpoints',
        'rotation': 'NumPy default_rng(seed) Gaussian QR, column signs corrected by diag(R)',
        'isotropic': 'Prior decoder outputs copied verbatim; energies reconstructed without baseline inference',
        'energy_units': 'Normalized by shared s_ref; MSE is per dimension; tail8 is summed squared PC projection',
        'actual_energy': 'Realized displacement after casting noisy decoder input to FP32',
        'constructed_energy': 'Float64 intended displacement; aligned/rotated preserve pointwise squared norm',
        'tail_axes': 'Lowest 8 eigenvectors of the shared reference covariance, common to all paths',
        'decoder': 'Existing perturb helper at sigma=0 reads preperturbed FP32 latent, target bytes plus true EOS',
        'scope': 'Frozen AE inference; no whitening, model fitting, architecture changes, or new training'},
        'models': {}, 'experiments': {}}
    q = haar_orthogonal(48, args.rotation_seed)
    np.save(args.output_dir / 'haar_rotation.npy', q)
    for model_tag, info in models.items():
        checkpoint = Path(info['checkpoint'])
        if digest(checkpoint) != info['checkpoint_sha256']:
            raise ValueError(f'Checkpoint changed: {checkpoint}')
        with np.load(info['npz'], allow_pickle=False) as bank:
            bank_hex, bank_z = bank['hex'].tolist(), bank['z'].copy()
        lookup = {hx: i for i, hx in enumerate(bank_hex)}
        ref = bank_z[[lookup[hx] for hx in reference_hex]]
        fit = covariance_shape(ref)
        scale = info['reference_scalar_scale']
        # Provenance guards on the reused reference, rather than a new probe.
        if not np.allclose(fit['mean'], info['reference_mean'], atol=1e-12, rtol=1e-12):
            raise ValueError('Saved reference mean disagrees with reused NPZ')
        if not np.isclose(fit['scalar_scale'], scale, atol=1e-12, rtol=1e-12):
            raise ValueError('Saved reference scale disagrees with reused NPZ')
        a, tail = fit['shape'], fit['axes'][:, :8]
        shape_cov = a @ a.T
        covariances = {'isotropic': np.eye(48), 'aligned': shape_cov,
                       'rotated': q @ shape_cov @ q.T}
        theoretical = {arm: {
            'trace': float(np.trace(covariance)),
            'mse_per_sigma_squared': float(np.trace(covariance) / 48),
            'pc_tail8_energy_per_sigma_squared': float(np.trace(tail.T @ covariance @ tail)),
            'pc_tail8_fraction_of_expected_energy': float(np.trace(tail.T @ covariance @ tail) / np.trace(covariance))}
            for arm, covariance in covariances.items()}
        np.savez_compressed(args.output_dir / f'reference_geometry_{model_tag}.npz',
                            **fit, haar_rotation=q)
        result['models'][model_tag] = {
            **{key: value for key, value in info.items() if key != 'npz'},
            'embedding_npz': str(info['npz']), 'embedding_npz_sha256': digest(info['npz']),
            'reference_eigenvalues_descending': fit['eigenvalues'][::-1].tolist(),
            'reference_negative_eigenvalues_clipped': int((fit['eigenvalues_raw'] < 0).sum()),
            'reference_tail8_variance_fraction': float(fit['eigenvalues'][:8].sum() / fit['trace']),
            'theoretical_noise': theoretical}
        saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
        ae = ByteStringAE(Cfg(saved['cfg'])).to(device=args.device, dtype=torch.float32).eval()
        ae.load_state_dict(saved['model'], strict=True)
        ae.requires_grad_(False)
        del saved
        for name, (directory, previous) in experiments.items():
            if previous['model_tag'] != model_tag:
                continue
            # The low-lambda run has its own saved target string union.
            if directory / f'embeddings_{model_tag}.npz' != info['npz']:
                with np.load(directory / f'embeddings_{model_tag}.npz', allow_pickle=False) as extra:
                    extra_lookup = dict(zip(extra['hex'].tolist(), extra['z']))
            else:
                extra_lookup = None
            source_file = directory / f'noise_{name}.jsonl.gz'
            selected = defaultdict(list)
            with gzip.open(source_file, 'rt', encoding='utf-8') as handle:
                for line in handle:
                    row = json.loads(line)
                    if row['sigma'] in args.sigmas:
                        selected[(row['sigma'], row['seed'])].append(row)
            group = {key: previous[key] for key in
                     ('model_tag', 'path_source_tag', 'variant', 'documents', 'units', 'bytes', 'noise_panel_units')}
            group['source_records'] = str(source_file)
            group['source_records_sha256'] = digest(source_file)
            group['arms'] = {arm: {'noise': {}, 'records': f'noise_{name}__{arm}.jsonl.gz'}
                             for arm in ('isotropic', 'aligned', 'rotated')}
            handles = {arm: gzip.open(args.output_dir / details['records'], 'wt', encoding='utf-8')
                       for arm, details in group['arms'].items()}
            max_energy_difference = 0.
            max_realized_energy_difference = 0.
            try:
                for sigma in args.sigmas:
                    all_records = {arm: [] for arm in handles}
                    for (saved_sigma, seed), baseline in selected.items():
                        if saved_sigma != sigma:
                            continue
                        samples = [{key: row[key] for key in SAMPLE_KEYS} for row in baseline]
                        weights = [row['weight'] for row in baseline]
                        z = np.asarray([extra_lookup[row['hex']] if extra_lookup is not None
                                        else bank_z[lookup[row['hex']]] for row in baseline], dtype=np.float32)
                        epsilon = epsilon_for(baseline, z.shape[1])
                        # Match original NumPy FP32 arithmetic for baseline realization.
                        isotropic_noisy = z + scale * sigma * epsilon
                        aligned = sigma * (epsilon.astype(np.float64) @ a.T)
                        rotated = aligned @ q.T
                        max_energy_difference = max(max_energy_difference, float(np.max(np.abs(
                            np.mean(aligned ** 2, axis=1) - np.mean(rotated ** 2, axis=1)))))
                        baseline_copy = [dict(row) for row in baseline]
                        append_energies(baseline_copy, sigma * epsilon.astype(np.float64),
                                        (isotropic_noisy.astype(np.float64) - z) / scale, tail)
                        all_records['isotropic'].extend(baseline_copy)
                        realized_mses = {}
                        for arm, delta in [('aligned', aligned), ('rotated', rotated)]:
                            noisy = (z.astype(np.float64) + scale * delta).astype(np.float32)
                            records = perturb(ae, noisy, samples, weights, scale, 0., seed,
                                              args.device, args.batch_size)
                            actual = (noisy.astype(np.float64) - z) / scale
                            append_energies(records, delta, actual, tail)
                            for row in records:
                                row['sigma'] = sigma
                                row['actual_noise_rmse'] = float(np.sqrt(row['actual_noise_mse']))
                            all_records[arm].extend(records)
                            realized_mses[arm] = np.mean(actual ** 2, axis=1)
                        max_realized_energy_difference = max(max_realized_energy_difference, float(np.max(
                            np.abs(realized_mses['aligned'] - realized_mses['rotated']))))
                    for arm, records in all_records.items():
                        if not records:
                            raise ValueError(f'Missing saved panel: {name}, sigma={sigma}')
                        write_rows(handles[arm], records)
                        group['arms'][arm]['noise'][str(sigma)] = grouped_with_energy(records)
                group['aligned_rotated_max_constructed_mse_difference'] = max_energy_difference
                group['aligned_rotated_max_realized_mse_difference'] = max_realized_energy_difference
            finally:
                for handle in handles.values():
                    handle.close()
            result['experiments'][name] = group
            write_json(args.output_dir / 'results.json', result)
            print(json.dumps({'event': 'group_complete', 'name': name,
                              'byte_em': {arm: {sigma: stats['overall']['em_byte_weighted']
                                  for sigma, stats in details['noise'].items()}
                                  for arm, details in group['arms'].items()},
                              'elapsed_seconds': time.perf_counter() - started}), flush=True)
        del ae
        if args.device.startswith('cuda'):
            torch.cuda.empty_cache()
    result['elapsed_seconds'] = time.perf_counter() - started
    write_json(args.output_dir / 'results.json', result)
    print(json.dumps({'event': 'completed', 'groups': len(result['experiments']),
                      'elapsed_seconds': result['elapsed_seconds']}), flush=True)


if __name__ == '__main__':
    main()
