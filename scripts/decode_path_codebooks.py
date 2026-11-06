"""Decode saved closed-codebook posterior means with the frozen AE decoder.

MAP readout uses the original bytes of the selected native feasible bank member.
The experiment has a finite closed prior and no text history or model training.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import gzip
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch

from analyze_candidate_risks import decoder_logits
from analyze_path_codebooks import file_sha256, write_json
from analyze_path_latents import edit_distance
from char_diff.vendor.bltz.config import Cfg
from char_diff.vendor.bltz.models.autoencoder import ByteStringAE


def load_posterior(path, model_info, expected_tag):
    with np.load(path, allow_pickle=False) as saved:
        data = {key: saved[key].copy() for key in saved.files}
    vectors = data['posterior_mean_normalized']
    n = len(vectors)
    reference_mean = np.asarray(model_info['reference_mean'])
    if vectors.shape != (n, len(reference_mean)) or not np.isfinite(vectors).all():
        raise ValueError(f'Invalid posterior mean matrix: {path}')
    for key in ['true_hex', 'map_hex', 't', 'seed', 'query_id', 'true_posterior_mass',
                'map_posterior_mass', 'mean_to_true_rmse_normalized']:
        if data[key].shape != (n,):
            raise ValueError(f'Expected one {key} per posterior mean: {path}')
    if str(data['coordinate_system']) != 'reference_standardized' or str(data['model_tag']) != expected_tag:
        raise ValueError(f'Posterior coordinates/model tag mismatch: {path}')
    if (not np.array_equal(data['reference_mean'], reference_mean)
            or float(data['reference_scalar_scale']) != model_info['reference_scalar_scale']
            or str(data['checkpoint_sha256']) != model_info['checkpoint_sha256']):
        raise ValueError(f'Posterior reference/checkpoint metadata mismatch: {path}')
    for key in ['true_hex', 'map_hex']:
        for hx in set(data[key]):
            bytes.fromhex(str(hx)).decode('utf-8', errors='strict')
    return data


def summarize(rows, prefix):
    lengths = np.array([r['bytes'] for r in rows], dtype=np.float64)
    exact = np.array([r[f'{prefix}_exact'] for r in rows], dtype=np.float64)
    return {'query_noise_pairs': len(rows), 'em_unit': float(exact.mean()),
            'em_byte_weighted': float(lengths @ exact / lengths.sum()),
            'byte_edit_rate': float(sum(r[f'{prefix}_byte_edit'] for r in rows) / lengths.sum()),
            'utf8_valid_fraction': float(np.mean([r[f'{prefix}_utf8_valid'] for r in rows])),
            'eos_present_fraction': float(np.mean([r[f'{prefix}_eos_present'] for r in rows])),
            'empty_fraction': float(np.mean([r[f'{prefix}_empty'] for r in rows]))}


def summarize_pair(rows):
    return {'posterior_mean_decoder': summarize(rows, 'mean'),
            'map_native_bytes': summarize(rows, 'map'),
            'mean_exact_given_map_correct': float(np.mean([r['mean_exact'] for r in rows if r['map_exact']]))
                if any(r['map_exact'] for r in rows) else None,
            'map_correct_mean_wrong': sum(r['map_exact'] and not r['mean_exact'] for r in rows),
            'map_wrong_mean_correct': sum(not r['map_exact'] and r['mean_exact'] for r in rows),
            'mean_to_true_rmse_normalized': float(np.mean([r['mean_to_true_rmse_normalized'] for r in rows]))}


@torch.inference_mode()
def decode_records(ae, data, device, batch_size):
    normalized = data['posterior_mean_normalized']
    raw = (normalized * float(data['reference_scalar_scale']) + data['reference_mean']).astype(np.float32)
    rows = []
    for start in range(0, len(raw), batch_size):
        logits = decoder_logits(ae, torch.as_tensor(raw[start:start + batch_size], device=device))
        if not torch.isfinite(logits).all():
            raise FloatingPointError('Non-finite posterior mean decoder logits')
        predictions = logits.argmax(-1).cpu().tolist()
        for offset, prediction in enumerate(predictions):
            i = start + offset
            target = bytes.fromhex(str(data['true_hex'][i]))
            mapped = bytes.fromhex(str(data['map_hex'][i]))
            eos = 256 in prediction
            decoded = bytes(prediction[:prediction.index(256)] if eos else prediction)
            try:
                decoded.decode('utf-8', errors='strict')
                utf8 = True
            except UnicodeDecodeError:
                utf8 = False
            rows.append({'true_hex': target.hex(), 'map_hex': mapped.hex(), 'mean_decoded_hex': decoded.hex(),
                         'bytes': len(target), 't': float(data['t'][i]), 'seed': int(data['seed'][i]),
                         'query_id': int(data['query_id'][i]),
                         'mean_exact': prediction[:len(target) + 1] == list(target) + [256],
                         'mean_byte_edit': edit_distance(target, decoded), 'mean_utf8_valid': utf8,
                         'mean_eos_present': eos, 'mean_empty': not decoded,
                         'map_exact': mapped == target, 'map_byte_edit': edit_distance(target, mapped),
                         'map_utf8_valid': True, 'map_eos_present': True, 'map_empty': not mapped,
                         'true_posterior_mass': float(data['true_posterior_mass'][i]),
                         'map_posterior_mass': float(data['map_posterior_mass'][i]),
                         'mean_to_true_rmse_normalized': float(data['mean_to_true_rmse_normalized'][i])})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--codebook-results', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--model-tags', nargs='*')
    parser.add_argument('--check-inputs', action='store_true', help='Read/validate NPZ shapes only; no model loading')
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error('batch-size must be positive')
    core = json.loads(args.codebook_results.read_text(encoding='utf-8'))
    jobs = []
    for name, experiment in core['experiments'].items():
        tag = experiment.get('model_tag')
        if args.model_tags and tag not in args.model_tags:
            continue
        for bank in experiment.get('banks', {}).values():
            artifact = bank.get('posterior_records')
            if artifact:
                jobs.append((name, tag, bank, artifact))
    if not jobs:
        raise ValueError('No saved primary-bank posteriors in codebook results')
    started = time.perf_counter()
    result = {'protocol': {'scope': 'frozen finite-codebook analytic posterior readout; no text history or training',
                           'posterior_mean': 'reference-standardized float64 means inverted to raw FP32 before decoder',
                           'mean_em': 'argmax target bytes plus true EOS; free readout stops at first EOS',
                           'map_readout': 'original bytes of MAP native feasible bank member; EOS feasibility inherited from path graph',
                           'weighting': 'equal query types and noise seeds; byte EM weights complete-unit hits by target bytes',
                           'device': args.device, 'batch_size': args.batch_size,
                           'codebook_results_sha256': file_sha256(args.codebook_results),
                           'script_sha256': file_sha256(__file__), 'check_inputs_only': args.check_inputs},
              'experiments': {}}
    ae, loaded_tag = None, None
    torch.set_num_threads(4)
    for name, tag, bank, artifact in jobs:
        info = core['models'][tag]
        path = Path(artifact['path'])
        data = load_posterior(path, info, tag)
        if args.check_inputs:
            result['experiments'][name] = {'rows': len(data['t']), 'dimension': data['posterior_mean_normalized'].shape[1],
                                           'status': 'parsed_shapes_valid'}
            continue
        if loaded_tag != tag:
            if ae is not None:
                del ae
            state = torch.load(info['checkpoint'], map_location='cpu', weights_only=False)
            ae = ByteStringAE(Cfg(state['cfg'])).to(device=args.device, dtype=torch.float32).eval()
            ae.load_state_dict(state['model'], strict=True)
            ae.requires_grad_(False)
            loaded_tag = tag
        rows = decode_records(ae, data, args.device, args.batch_size)
        by_time = defaultdict(list)
        for row in rows:
            by_time[str(row['t'])].append(row)
        experiment = {'model_tag': tag, 'variant': core['experiments'][name]['variant'],
                      'bank_size': bank['bank_size'], 'query_count': bank['query_count'],
                      'checkpoint_sha256': info['checkpoint_sha256'], 'posterior_sha256': file_sha256(path),
                      'times': {t: {**summarize_pair(part), 'by_seed': {
                          str(seed): summarize_pair([r for r in part if r['seed'] == seed])
                          for seed in sorted({r['seed'] for r in part})}}
                          for t, part in sorted(by_time.items(), key=lambda item: float(item[0]))}}
        args.output_dir.mkdir(parents=True, exist_ok=True)
        with gzip.open(args.output_dir / f'decoded_{name}.jsonl.gz', 'wt', encoding='utf-8') as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
        result['experiments'][name] = experiment
        result['elapsed_seconds'] = time.perf_counter() - started
        write_json(args.output_dir / 'results.json', result)
        print(json.dumps({'event': 'codebook_readout_complete', 'name': name, 'rows': len(rows),
                          'elapsed_seconds': result['elapsed_seconds']}), flush=True)
    result['elapsed_seconds'] = time.perf_counter() - started
    write_json(args.output_dir / ('input_check.json' if args.check_inputs else 'results.json'), result)
    if args.check_inputs:
        print(json.dumps({'event': 'input_check_passed', 'groups': len(jobs), 'elapsed_seconds': result['elapsed_seconds']}), flush=True)


if __name__ == '__main__':
    main()
