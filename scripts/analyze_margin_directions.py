"""Frozen decoder margin gradients against shared reference noise directions."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch

from scripts.analyze_candidate_risks import (
    ByteStringAE, Cfg, decoder_logits, encode, read_rows, sha,
    targets_from_lengths, tensorize, write_json,
)


def direction_variances(gradients, shape, rotation, axes):
    """Gradients are rows; column perturbations are A eps and Q A eps."""
    g = np.asarray(gradients, dtype=np.float64)
    return {
        'isotropic': np.sum(g * g, axis=1),
        'aligned': np.sum((g @ shape) ** 2, axis=1),
        'rotated': np.sum((g @ rotation @ shape) ** 2, axis=1),
        'tail8_energy': np.sum((g @ axes[:, :8]) ** 2, axis=1),
    }


def flip_probability(margin, variance, sigma):
    if variance <= 0:
        return float(margin < 0)
    return .5 * math.erfc(margin / (sigma * math.sqrt(2 * variance)))


def linear_formula_check():
    """One 48D known linear margin, exact ±basis probes, and known rotation."""
    d = 48
    expected_g = np.linspace(.2, 1.7, d)
    z = torch.zeros(d, dtype=torch.float64, requires_grad=True)
    linear_margin = 1.25 + (z * torch.tensor(expected_g)).sum()
    g = torch.autograd.grad(linear_margin, z)[0].detach().numpy()
    values = np.linspace(.1, 2., d)
    values *= d / values.sum()
    a = np.diag(np.sqrt(values))
    q = np.eye(d)[:, ::-1]
    axes = np.eye(d)
    result = direction_variances(g[None], a, q, axes)
    probes = np.concatenate((np.sqrt(d) * np.eye(d), -np.sqrt(d) * np.eye(d)))
    np.testing.assert_allclose(g, expected_g, atol=0, rtol=0)
    for arm, transform in [('isotropic', np.eye(d)), ('aligned', a), ('rotated', q @ a)]:
        changes = probes @ transform.T @ g
        np.testing.assert_allclose(result[arm][0], np.mean(changes ** 2), rtol=1e-14)
    np.testing.assert_allclose(result['tail8_energy'][0], np.sum(g[:8] ** 2), rtol=1e-14)
    np.testing.assert_allclose(flip_probability(1., 4., .5), .15865525393145707, rtol=1e-14)
    return {'name': 'known_48d_linear_gradient_direction_variances_and_flip', 'passed': True,
            'dimension': d, 'isotropic_variance': float(result['isotropic'][0]),
            'aligned_ratio': float(result['aligned'][0] / result['isotropic'][0]),
            'rotated_ratio': float(result['rotated'][0] / result['isotropic'][0])}


def margin_gradient_vectors(ae, latent, targets):
    """Choose weakest raw margin and its runner-up at the clean latent."""
    with torch.enable_grad():
        z = latent.detach().clone().requires_grad_(True)
        logits = decoder_logits(ae, z)
        valid = targets != -100
        safe = targets.clamp_min(0)
        runner = logits.detach().clone().scatter(-1, safe[..., None], -math.inf).argmax(-1)
        margins = logits.gather(-1, safe[..., None]).squeeze(-1) - logits.gather(-1, runner[..., None]).squeeze(-1)
        weakest = margins.masked_fill(~valid, math.inf).argmin(-1)
        indices = torch.arange(len(z), device=z.device)
        selected = margins[indices, weakest]
        gradient = torch.autograd.grad(selected.sum(), z, create_graph=False)[0]
        exact = ((logits.argmax(-1) == targets) | ~valid).all(-1)
        if not bool(exact.all()):
            raise ValueError('Previously feasible candidate failed clean reconstruction')
        if not bool(torch.isfinite(gradient).all() and torch.isfinite(selected).all()):
            raise FloatingPointError('Nonfinite clean margin or gradient')
        return {
            'gradient_raw': gradient.detach().cpu().numpy(),
            'margin': selected.detach().cpu().numpy(),
            'position': weakest.detach().cpu().numpy(),
            'runnerup_class': runner[indices, weakest].detach().cpu().numpy(),
            'correct_class': safe[indices, weakest].detach().cpu().numpy(),
        }


def distribution(values):
    a = np.asarray([v for v in values if v is not None], dtype=np.float64)
    if not len(a):
        return {'count': 0}
    return {'count': len(a), 'mean': float(a.mean()), 'median': float(np.median(a)),
            'quantiles': {str(q): float(np.quantile(a, q)) for q in [.05, .25, .75, .95]},
            'min': float(a.min()), 'max': float(a.max())}


def summarize(rows, sigmas):
    result = {'candidate_count': len(rows),
              'language_counts': dict(sorted(Counter(r['language'] for r in rows).items())),
              'mean_byte_length': sum(r['bytes'] for r in rows) / len(rows),
              'flat_gradient_candidates': sum(r['variances']['isotropic'] == 0 for r in rows),
              'metrics': {name: distribution([r[name] for r in rows]) for name in
                          ['margin', 'aligned_variance_ratio', 'rotated_variance_ratio', 'tail8_gradient_energy_fraction']},
              'ratios_above_one': {arm: sum(r[f'{arm}_variance_ratio'] is not None and r[f'{arm}_variance_ratio'] > 1 for r in rows) / len(rows)
                                   for arm in ['aligned', 'rotated']},
              'linear_flip_probability': {str(sigma): {arm: distribution([r['linear_flip_probability'][str(sigma)][arm] for r in rows])
                                                       for arm in ['isotropic', 'aligned', 'rotated']} for sigma in sigmas}}
    total_energy = sum(r['variances']['isotropic'] for r in rows)
    result['pooled_tail8_gradient_energy_fraction'] = sum(r['variances']['tail8_energy'] for r in rows) / total_energy if total_energy else None
    return result


def run(args):
    start = time.perf_counter()
    torch.set_num_threads(4)
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    if any((output / name).exists() for name in ['rows_4g.jsonl', 'rows_8g.jsonl', 'results.json']):
        raise FileExistsError('Use a new output directory')
    core_path = args.base_dir / 'latents/results.json'
    core = json.loads(core_path.read_text(encoding='utf-8'))
    source_rows, source_metadata, source_paths = {}, {}, {}
    for tag in ['4g', '8g']:
        path = args.base_dir / f'risk_{tag}/candidate_risks.jsonl'
        source_paths[tag] = path
        source_rows[tag] = {(r['doc_id'], r['unit_hex']): r for r in read_rows(path)
                            if r.get('fold') == 'analysis' and r['feasible']}
        source_metadata[tag] = json.loads((path.parent / 'results.json').read_text(encoding='utf-8'))
    common = set(source_rows['4g']) & set(source_rows['8g'])
    selected = sorted(common, key=lambda key: hashlib.sha256(('margin-direction-v1:' + key[0] + ':' + key[1]).encode()).digest())[:args.max_candidates]
    if not selected:
        raise ValueError('No common feasible candidates')
    selection_digest = hashlib.sha256(json.dumps(selected, separators=(',', ':')).encode()).hexdigest()
    result = {'protocol': {
        'core_results': str(core_path), 'core_results_sha256': sha(core_path),
        'script_sha256': sha(__file__), 'common_available_candidates': len(common),
        'candidate_count': len(selected), 'selected_keys_sha256': selection_digest,
        'selection': 'lowest SHA256(margin-direction-v1:doc_id:unit_hex), among common clean-feasible analysis candidates',
        'same_selected_keys_across_checkpoints': True, 'sigmas': args.sigmas, 'device': args.device, 'precision': 'fp32 model and gradient; float64 covariance algebra',
        'gradient': 'g=s_ref*grad_z m for weakest clean correct-versus-runner-up raw-logit margin; selected position and runner-up fixed locally',
        'aligned_variance': '||A^T g||^2, relative to isotropic ||g||^2',
        'rotated_variance': '||A^T Q^T g||^2; exact same saved Q and A as covariance-noise ablation',
        'tail8': 'fraction of gradient squared norm in eight lowest reference covariance eigenvectors',
        'linear_flip': 'Phi(-m/(sigma*sqrt(variance))); a single local linear constraint, without multi-position union or curvature',
        'scope': 'no parameter updates; model frozen; only detached latent requires gradient'}, 'models': {}}
    for tag in ['4g', '8g']:
        model_start = time.perf_counter()
        info = core['models'][tag]
        checkpoint = Path(info['checkpoint'])
        checkpoint_hash = sha(checkpoint)
        if checkpoint_hash != info['checkpoint_sha256'] or checkpoint_hash != source_metadata[tag]['checkpoint_sha256']:
            raise ValueError('Core/risk checkpoint identity mismatch')
        geometry_path = args.base_dir / f'covariance_noise/reference_geometry_{tag}.npz'
        with np.load(geometry_path, allow_pickle=False) as geometry:
            shape, rotation, axes = geometry['shape'].copy(), geometry['haar_rotation'].copy(), geometry['axes'].copy()
            scale = float(geometry['scalar_scale'])
        if not np.isclose(scale, info['reference_scalar_scale'], rtol=1e-12, atol=1e-12):
            raise ValueError('Geometry and core reference scales differ')
        saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
        ae = ByteStringAE(Cfg(saved['cfg'])).to(device=args.device, dtype=torch.float32).eval()
        ae.load_state_dict(saved['model'], strict=True)
        ae.requires_grad_(False)
        del saved
        selected_rows = [source_rows[tag][key] for key in selected]
        strings = [bytes.fromhex(r['unit_hex']) for r in selected_rows]
        z = encode(ae, strings, args.device, args.batch_size)
        batches = []
        for offset in range(0, len(strings), args.batch_size):
            part = strings[offset:offset + args.batch_size]
            ids, lengths, _ = tensorize(part, ae.l_max, args.device)
            targets = targets_from_lengths(ids, lengths)
            batches.append(margin_gradient_vectors(ae, torch.tensor(z[offset:offset + len(part)], device=args.device), targets))
        details = {key: np.concatenate([batch[key] for batch in batches]) for key in batches[0]}
        gradients = scale * details['gradient_raw'].astype(np.float64)
        variances = direction_variances(gradients, shape, rotation, axes)
        rows = []
        for i, original in enumerate(selected_rows):
            variance = {key: float(value[i]) for key, value in variances.items()}
            base = variance['isotropic']
            margin = float(details['margin'][i])
            row = {key: original[key] for key in ['doc_id', 'unit_hex', 'language', 'fold', 'bytes']}
            row.update({'checkpoint_tag': tag, 'margin': margin, 'weak_position': int(details['position'][i]),
                        'correct_class': int(details['correct_class'][i]), 'runnerup_class': int(details['runnerup_class'][i]),
                        'variances': variance, 'aligned_variance_ratio': variance['aligned'] / base if base else None,
                        'rotated_variance_ratio': variance['rotated'] / base if base else None,
                        'tail8_gradient_energy_fraction': variance['tail8_energy'] / base if base else None,
                        'linear_flip_probability': {str(sigma): {arm: flip_probability(margin, variance[arm], sigma)
                                                                  for arm in ['isotropic', 'aligned', 'rotated']} for sigma in args.sigmas}})
            rows.append(row)
        assert all(param.grad is None and not param.requires_grad for param in ae.parameters())
        with (output / f'rows_{tag}.jsonl').open('w', encoding='utf-8') as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False, separators=(',', ':')) + '\n')
        np.savez_compressed(output / f'gradients_{tag}.npz', g_standardized=gradients,
                            gradient_raw=details['gradient_raw'], z=z, margin=details['margin'],
                            weak_position=details['position'], correct_class=details['correct_class'], runnerup_class=details['runnerup_class'],
                            doc_id=np.asarray([key[0] for key in selected]), unit_hex=np.asarray([key[1] for key in selected]))
        result['models'][tag] = {**summarize(rows, args.sigmas), 'checkpoint': str(checkpoint),
                                'checkpoint_sha256': checkpoint_hash, 'reference_geometry_sha256': sha(geometry_path),
                                'candidate_rows_sha256': sha(source_paths[tag]), 'reference_scalar_scale': scale,
                                'parameter_gradients_absent': True, 'elapsed_seconds': time.perf_counter() - model_start}
        del ae
        write_json(output / 'results.json', result)
        print(json.dumps({'completed_model': tag, 'candidates': len(rows), 'seconds': result['models'][tag]['elapsed_seconds']}), flush=True)
    result['elapsed_seconds'] = time.perf_counter() - start
    write_json(output / 'results.json', result)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base-dir', type=Path, default=Path('runs/path_diffusability_20261104'))
    p.add_argument('--output-dir', type=Path)
    p.add_argument('--max-candidates', type=int, default=512)
    p.add_argument('--batch-size', type=int, default=256)
    p.add_argument('--sigmas', type=float, nargs='+', default=[.025, .05])
    p.add_argument('--device', default='cuda')
    p.add_argument('--check-only', action='store_true')
    args = p.parse_args()
    if args.output_dir is None:
        args.output_dir = args.base_dir / 'covariance_noise/margin_directions'
    if args.max_candidates <= 0 or args.batch_size <= 0 or any(sigma <= 0 for sigma in args.sigmas):
        p.error('Candidate cap, batch size, and sigma must be positive')
    if args.check_only:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        check = linear_formula_check()
        write_json(args.output_dir / 'math_check.json', {'check': check, 'script_sha256': sha(__file__)})
        print(json.dumps(check), flush=True)
    else:
        run(args)


if __name__ == '__main__':
    main()
