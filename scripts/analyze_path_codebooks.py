"""Closed-codebook Bayes diagnostics using saved frozen-AE embeddings only.

Targets belong to the finite bank by definition. This measures distinguishability
of distinct unit types under Gaussian corruption, without a fitted predictor or
text history. All paths share the core analysis reference mean/scalar scale.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import numpy as np


DEFAULT_VARIANTS = ['min_k', 'nll_1', 'nll_10', 'nll_1_bpe_0.5_samek']
SENSITIVITY_VARIANTS = {'min_k', 'nll_1'}


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n',
                    encoding='utf-8')


def ranked_hex(strings, domain):
    prefix = domain.encode('ascii') + b'\0'
    return sorted(strings, key=lambda hx: (hashlib.sha256(prefix + bytes.fromhex(hx)).digest(), hx))


def ordered_set_hash(strings):
    return hashlib.sha256(('\n'.join(strings) + '\n').encode('ascii')).hexdigest()


def read_groups(paths, fold, variants):
    groups = {}
    seen_documents = set()
    for path in paths:
        with Path(path).open(encoding='utf-8') as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get('fold', 'analysis') != fold or row['variant'] not in variants:
                    continue
                key = (row['checkpoint_tag'], row['variant'])
                document_key = (*key, str(row['doc_id']))
                if document_key in seen_documents:
                    raise ValueError(f'Duplicate document/variant input: {document_key}')
                seen_documents.add(document_key)
                group = groups.setdefault(key, {'types': {}, 'documents': 0, 'covered_documents': 0,
                                                'source_bytes': 0, 'occurrences': 0})
                group['documents'] += 1
                group['source_bytes'] += row.get('source_bytes', 0)
                if not row.get('units'):
                    continue
                group['covered_documents'] += 1
                language = row['language']
                for unit in row['units']:
                    hx = bytes.fromhex(unit.get('hex', unit.get('unit_hex'))).hex()
                    value = group['types'].setdefault(hx, {'occurrences': 0, 'languages': Counter()})
                    value['occurrences'] += 1
                    value['languages'][language] += 1
                    group['occurrences'] += 1
    return groups


def composition(strings, types):
    lengths = Counter()
    language_presence = Counter()
    language_fraction = Counter()
    occurrences_by_language = Counter()
    for hx in strings:
        lengths[len(bytes.fromhex(hx))] += 1
        languages = types[hx]['languages']
        for language, count in languages.items():
            language_presence[language] += 1
            language_fraction[language] += 1 / len(languages)
            occurrences_by_language[language] += count
    return {'types': len(strings),
            'mean_byte_length': sum(k * v for k, v in lengths.items()) / len(strings) if strings else None,
            'types_by_byte_length': {str(k): v for k, v in sorted(lengths.items())},
            'type_language_presence_overlapping': dict(sorted(language_presence.items())),
            'type_language_equal_share': dict(sorted(language_fraction.items())),
            'source_occurrences_by_language': dict(sorted(occurrences_by_language.items()))}


def unit_noise(strings, dimension, seed):
    result = []
    for hx in strings:
        material = b'codebook-noise-v1\0' + str(seed).encode('ascii') + b'\0' + bytes.fromhex(hx)
        value = int.from_bytes(hashlib.sha256(material).digest()[:16], 'little')
        result.append(np.random.default_rng(value).standard_normal(dimension))
    return np.asarray(result, dtype=np.float64)


def posterior(bank, observed, t):
    """Exact posterior for a finite, uniform bank and Xt=(1-t)X+t*epsilon."""
    bank = np.asarray(bank, dtype=np.float64)
    observed = np.asarray(observed, dtype=np.float64)
    if not 0 < t <= 1:
        raise ValueError('Posterior requires 0 < t <= 1')
    means = (1 - t) * bank
    squared = (np.sum(observed ** 2, axis=1, keepdims=True)
               + np.sum(means ** 2, axis=1)[None, :] - 2 * observed @ means.T)
    logw = -np.maximum(squared, 0) / (2 * t ** 2)
    logw -= np.max(logw, axis=1, keepdims=True)
    logw -= np.log(np.exp(logw).sum(axis=1, keepdims=True))
    weights = np.exp(logw)
    return weights @ bank, weights, -np.sum(weights * logw, axis=1)


def summarize_metrics(squared_error, entropy, top1, mass, trace, bank_size):
    mse = float(np.mean(squared_error))
    return {'mse_squared_l2': mse,
            'normalized_mse_over_trace': mse / trace if trace > 0 else None,
            'posterior_entropy_nats': float(np.mean(entropy)),
            'posterior_entropy_over_log_n': float(np.mean(entropy)) / math.log(bank_size) if bank_size > 1 else None,
            'top1_type_accuracy': float(np.mean(top1)),
            'correct_type_posterior_mass': float(np.mean(mass)),
            'query_noise_pairs': len(squared_error)}


def evaluate_bank(bank, strings, query_strings, times, seeds, query_batch, *, posterior_records=None):
    bank = np.asarray(bank, dtype=np.float64)
    lookup = {hx: i for i, hx in enumerate(strings)}
    target_ids = np.asarray([lookup[hx] for hx in query_strings], dtype=np.int64)
    targets = bank[target_ids]
    mean = bank.mean(0)
    trace = float(np.mean(np.sum((bank - mean) ** 2, axis=1)))
    result = {'bank_size': len(bank), 'dimension': bank.shape[1], 'bank_variance_trace': trace,
              'bank_mean': mean.tolist(), 'query_count': len(target_ids),
              'query_mean_squared_distance_to_bank_mean': float(np.mean(np.sum((targets - mean) ** 2, axis=1))),
              'unique_embedding_vectors': len(np.unique(bank, axis=0)), 'times': {}}
    noise = {seed: unit_noise(query_strings, bank.shape[1], seed) for seed in seeds}
    saved_records = defaultdict(list) if posterior_records is not None else None
    for t in times:
        per_seed, collected = {}, []
        for seed in seeds:
            parts = []
            for start in range(0, len(targets), query_batch):
                stop = min(start + query_batch, len(targets))
                clean = targets[start:stop]
                observed = (1 - t) * clean + t * noise[seed][start:stop]
                prediction, weights, entropy = posterior(bank, observed, t)
                ids = target_ids[start:stop]
                if saved_records is not None:
                    map_ids = weights.argmax(axis=1)
                    saved_records['posterior_mean_normalized'].append(prediction.copy())
                    saved_records['true_hex'].append(np.asarray(query_strings[start:stop]))
                    saved_records['map_hex'].append(np.asarray([strings[index] for index in map_ids]))
                    saved_records['t'].append(np.full(stop - start, t, dtype=np.float64))
                    saved_records['seed'].append(np.full(stop - start, seed, dtype=np.int64))
                    saved_records['query_id'].append(np.arange(start, stop, dtype=np.int64))
                    saved_records['true_bank_id'].append(ids.copy())
                    saved_records['map_bank_id'].append(map_ids)
                    saved_records['true_posterior_mass'].append(weights[np.arange(stop - start), ids])
                    saved_records['map_posterior_mass'].append(weights[np.arange(stop - start), map_ids])
                    saved_records['mean_to_true_rmse_normalized'].append(np.sqrt(np.mean((prediction - clean) ** 2, axis=1)))
                parts.append(np.column_stack((np.sum((prediction - clean) ** 2, axis=1), entropy,
                                              weights.argmax(axis=1) == ids,
                                              weights[np.arange(stop - start), ids])))
            values = np.concatenate(parts)
            collected.append(values)
            per_seed[str(seed)] = summarize_metrics(*values.T, trace, len(bank))
        values = np.concatenate(collected)
        result['times'][str(t)] = {'overall': summarize_metrics(*values.T, trace, len(bank)),
                                  'by_seed': per_seed}
    if saved_records is not None:
        posterior_records.update({key: np.concatenate(value) for key, value in saved_records.items()})
    return result


def synthetic_check():
    bank = np.array([[-1., 0.], [1., 0.]])
    mean, weights, entropy = posterior(bank, np.array([[.75, 2.], [-.75, -3.]]), 1.)
    np.testing.assert_allclose(mean, [[0., 0.], [0., 0.]], atol=1e-12)
    np.testing.assert_allclose(weights, .5, atol=1e-12)
    np.testing.assert_allclose(entropy, math.log(2), atol=1e-12)
    t, observed = .5, np.array([[.25, 0.]])
    mean, weights, _ = posterior(bank, observed, t)
    expected_positive_mass = 1 / (1 + math.exp(-2 * (1 - t) * observed[0, 0] / t ** 2))
    np.testing.assert_allclose(weights[0, 1], expected_positive_mass, atol=1e-12)
    np.testing.assert_allclose(mean[0, 0], 2 * expected_positive_mass - 1, atol=1e-12)
    return {'status': 'passed', 'case': 'two_point_analytic_posterior_and_t1_uniform_mean',
            't1_entropy': float(entropy[0]), 'analytic_positive_mass': expected_positive_mass}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--paths', type=Path, nargs='+')
    parser.add_argument('--latent-results', type=Path, help='Core results.json with reference mean and scalar scale')
    parser.add_argument('--embeddings-dir', type=Path, help='Default: parent directory of latent-results')
    parser.add_argument('--model-tags', nargs='*', help='Default: all native path checkpoint tags present in core results')
    parser.add_argument('--fold', default='analysis')
    parser.add_argument('--variants', nargs='+', default=DEFAULT_VARIANTS)
    parser.add_argument('--bank-size', type=int, default=2048)
    parser.add_argument('--sensitivity-size', type=int, default=1024)
    parser.add_argument('--query-count', type=int, default=256)
    parser.add_argument('--times', nargs='+', type=float, default=[.1, .25, .5, .75, .9])
    parser.add_argument('--seeds', nargs='+', type=int, default=[42, 123])
    parser.add_argument('--query-batch', type=int, default=64)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--save-posterior-dir', type=Path,
                        help='Optionally save primary-bank posterior means/labels in NPZ; normalized coordinates')
    parser.add_argument('--self-test', action='store_true', help='Only run the small analytic formula check')
    args = parser.parse_args()
    if args.self_test:
        result = synthetic_check()
        if args.output:
            write_json(args.output, result)
        print(json.dumps(result), flush=True)
        return
    if not args.paths or not args.latent_results or not args.output:
        parser.error('--paths, --latent-results, and --output are required for analysis')
    if min(args.bank_size, args.sensitivity_size, args.query_count, args.query_batch) < 1:
        parser.error('All sizes must be positive')
    if args.sensitivity_size > args.bank_size:
        parser.error('sensitivity-size must be at most bank-size')
    if not all(0 < t <= 1 for t in args.times):
        parser.error('times must satisfy 0 < t <= 1')
    started = time.perf_counter()
    core = json.loads(args.latent_results.read_text(encoding='utf-8'))
    embeddings_dir = args.embeddings_dir or args.latent_results.parent
    groups = read_groups(args.paths, args.fold, args.variants)
    tags = args.model_tags or sorted({tag for tag, _ in groups if tag in core['models']})
    if not tags:
        raise ValueError('No matching native checkpoint tags in path records and core results')
    result = {'protocol': {
        'scope': 'closed finite codebook; true targets are bank members; no training or text history',
        'prior': 'uniform over distinct byte strings; type-weighted, not occurrence-weighted',
        'distribution': 'native path analysis documents; unpaired lengths/languages across path families',
        'fold': args.fold, 'variants': args.variants, 'model_tags': tags,
        'bank_size_requested': args.bank_size, 'sensitivity_size_requested': args.sensitivity_size,
        'sensitivity_variants': sorted(SENSITIVITY_VARIANTS), 'query_count_requested': args.query_count,
        'bank_selection': 'ascending SHA256(codebook-bank-v1 NUL bytes), tie-break hex; nested banks',
        'query_selection': 'ascending SHA256(codebook-query-v1 NUL bytes) from smallest selected bank; same queries for paired bank sizes',
        'noise': 'unit-byte and seed keyed iid Gaussian; common across times/checkpoints/paths for shared strings',
        'times': args.times, 'seeds': args.seeds,
        'flow': 'Xt=(1-t)z+t*epsilon; logw_i=-||Xt-(1-t)z_i||^2/(2*t^2)',
        'normalization': '(raw_z-reference_mean)/reference_scalar_scale, shared across paths per checkpoint',
        'risk': 'mean query squared L2 error / population trace(Cov[bank]); Monte Carlo query/noise estimate',
        'entropy': 'mean posterior entropy / log(actual bank N)',
        'top1_ties': 'first member in deterministic bank order',
        'language_summary': 'presence overlaps; equal-share gives each type total language weight 1',
        'precision': 'numpy float64, batched over queries', 'query_batch': args.query_batch,
        'saved_posterior': 'primary bank only, reference-standardized float64 means; inverse transform stored in NPZ'
                           if args.save_posterior_dir else None,
        'paths_sha256': {str(p.resolve()): file_sha256(p) for p in args.paths},
        'latent_results_sha256': file_sha256(args.latent_results),
        'script_sha256': file_sha256(__file__), 'numpy_version': np.__version__,
        'python_version': sys.version}, 'models': {}, 'experiments': {}}
    for tag in tags:
        info = core['models'][tag]
        path = embeddings_dir / f'embeddings_{tag}.npz'
        with np.load(path, allow_pickle=False) as saved:
            strings = [hx.decode('ascii') if isinstance(hx, bytes) else str(hx) for hx in saved['hex']]
            vectors = np.asarray(saved['z'], dtype=np.float64)
        mean = np.asarray(info['reference_mean'], dtype=np.float64)
        scale = float(info['reference_scalar_scale'])
        if vectors.ndim != 2 or vectors.shape != (len(strings), len(mean)) or not np.isfinite(vectors).all():
            raise ValueError(f'Invalid embedding matrix for {tag}')
        if not math.isfinite(scale) or scale <= 0 or not np.isfinite(mean).all():
            raise ValueError(f'Invalid reference normalization for {tag}')
        lookup = {hx: i for i, hx in enumerate(strings)}
        if len(lookup) != len(strings):
            raise ValueError(f'Duplicate saved embedding keys for {tag}')
        result['models'][tag] = {**info, 'embeddings_path': str(path.resolve()),
                                 'embeddings_sha256': file_sha256(path)}
        for variant in args.variants:
            group = groups.get((tag, variant))
            group_name = f'{tag}__{variant}'
            if not group or not group['types']:
                result['experiments'][group_name] = {'status': 'no_covered_units'}
                continue
            types = group['types']
            ranked = ranked_hex(types, 'codebook-bank-v1')
            main_size = min(args.bank_size, len(ranked))
            sizes = [main_size]
            if variant in SENSITIVITY_VARIANTS:
                smaller = min(args.sensitivity_size, len(ranked))
                if smaller != main_size:
                    sizes.append(smaller)
            query_strings = ranked_hex(ranked[:min(sizes)], 'codebook-query-v1')[:args.query_count]
            experiment = {'status': 'complete', 'model_tag': tag, 'path_source_tag': tag, 'variant': variant,
                          **{key: value for key, value in group.items() if key != 'types'},
                          'eligible_unique_types': len(types), 'eligible_composition': composition(list(types), types),
                          'query_hex': query_strings, 'query_sha256': ordered_set_hash(query_strings),
                          'query_composition': composition(query_strings, types), 'banks': {}}
            for size in sizes:
                bank_strings = ranked[:size]
                missing = [hx for hx in bank_strings if hx not in lookup]
                if missing:
                    raise ValueError(f'{group_name}: {len(missing)} bank units absent from saved embeddings')
                bank = (vectors[[lookup[hx] for hx in bank_strings]] - mean) / scale
                posterior_records = {} if args.save_posterior_dir and size == main_size else None
                measured = evaluate_bank(bank, bank_strings, query_strings, args.times, args.seeds, args.query_batch,
                                         posterior_records=posterior_records)
                if posterior_records is not None:
                    args.save_posterior_dir.mkdir(parents=True, exist_ok=True)
                    posterior_path = args.save_posterior_dir / f'posterior_{group_name}.npz'
                    np.savez_compressed(posterior_path, **posterior_records, reference_mean=mean,
                                        reference_scalar_scale=np.asarray(scale), model_tag=np.asarray(tag),
                                        variant=np.asarray(variant), bank_size=np.asarray(size),
                                        coordinate_system=np.asarray('reference_standardized'),
                                        bank_sha256=np.asarray(ordered_set_hash(bank_strings)),
                                        checkpoint_sha256=np.asarray(info['checkpoint_sha256']))
                    measured['posterior_records'] = {'path': str(posterior_path.resolve()),
                                                     'sha256': file_sha256(posterior_path),
                                                     'rows': len(posterior_records['t']),
                                                     'coordinates': 'reference_standardized', 'dtype': 'float64'}
                experiment['banks'][str(size)] = {**measured, 'bank_hex': bank_strings,
                                                   'bank_sha256': ordered_set_hash(bank_strings),
                                                   'composition': composition(bank_strings, types)}
            result['experiments'][group_name] = experiment
            result['elapsed_seconds'] = time.perf_counter() - started
            write_json(args.output, result)
            print(json.dumps({'event': 'codebook_complete', 'name': group_name, 'sizes': sizes,
                              'queries': len(query_strings), 'elapsed_seconds': result['elapsed_seconds']}), flush=True)
    result['elapsed_seconds'] = time.perf_counter() - started
    write_json(args.output, result)


if __name__ == '__main__':
    main()
