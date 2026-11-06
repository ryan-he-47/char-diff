"""Does clean softmax quality predict latent-noise failures on unselected candidates?"""
from __future__ import annotations

import argparse
from collections import defaultdict
import gzip
import hashlib
import json
import math
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch

from char_diff.ae_quality import tensorize, targets_from_lengths
from char_diff.utils import write_json
from char_diff.vendor.bltz.config import Cfg
from char_diff.vendor.bltz.models.autoencoder import ByteStringAE


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_rows(path):
    opener = gzip.open if str(path).endswith('.gz') else open
    with opener(path, 'rt', encoding='utf-8') as handle:
        return [json.loads(line) for line in handle if line.strip()]


def weighted_auc(scores, positives, negatives):
    """AUROC with tied scores and integer replicate counts, high score=risk."""
    groups = defaultdict(lambda: [0, 0])
    for score, pos, neg in zip(scores, positives, negatives):
        if not math.isfinite(score):
            raise ValueError('AUC score must be finite')
        groups[float(score)][0] += int(pos)
        groups[float(score)][1] += int(neg)
    concordant = 0.
    negative_below = 0
    for score in sorted(groups):
        pos, neg = groups[score]
        concordant += pos * (negative_below + .5 * neg)
        negative_below += neg
    positive = sum(positives)
    negative = sum(negatives)
    pairs = positive * negative
    return {'auc': concordant / pairs if pairs else None, 'concordant_pairs': concordant,
            'positive_negative_pairs': pairs, 'positive_trials': positive, 'negative_trials': negative}


def summarize_auc(rows, key, sigma):
    def counts(part):
        pos = [row['failures'][sigma]['count'] for row in part]
        neg = [row['failures'][sigma]['trials'] - p for row, p in zip(part, pos)]
        return weighted_auc([row['risk_scores'][key] for row in part], pos, neg)
    groups = defaultdict(list)
    for row in rows:
        groups[(row['language'], row['bytes'])].append(row)
    parts = [counts(part) for part in groups.values()]
    numerator = sum(part['concordant_pairs'] for part in parts)
    denominator = sum(part['positive_negative_pairs'] for part in parts)
    return {'global': counts(rows), 'same_language_and_exact_byte_length': {
        'auc': numerator / denominator if denominator else None,
        'concordant_pairs': numerator, 'positive_negative_pairs': denominator,
        'informative_strata': sum(part['positive_negative_pairs'] > 0 for part in parts),
        'strata_total': len(parts)}}


@torch.no_grad()
def encode(ae, strings, device, batch_size):
    result = []
    for start in range(0, len(strings), batch_size):
        ids, _, mask = tensorize(strings[start:start+batch_size], ae.l_max, device)
        result.append(ae.encoder(ids, mask).float().cpu().numpy())
    return np.concatenate(result)


def decoder_logits(ae, z):
    k = ae.l_max + 1
    return ae.decoder(z.repeat_interleave(k, 0), torch.arange(k, device=z.device).repeat(len(z))).view(len(z), k, 257)


def weakest_margin_gradient(ae, z, targets, scale):
    """Piecewise local radius for the weakest clean-margin constraint only.

    r=m/(s_ref*||grad_z m||) is an L2 distance in standardized coordinates.
    For per-coordinate Gaussian sigma, this single linearized constraint flips
    with probability Phi(-r/sigma). Other constraints and curvature are separate.
    """
    with torch.enable_grad():
        z = z.detach().clone().requires_grad_(True)
        logits = decoder_logits(ae, z)
        valid = targets != -100
        safe = targets.clamp_min(0)
        runner = logits.detach().clone().scatter(-1, safe[..., None], -math.inf).argmax(-1)
        margins = logits.gather(-1, safe[..., None]).squeeze(-1) - logits.gather(-1, runner[..., None]).squeeze(-1)
        weakest = margins.masked_fill(~valid, math.inf).argmin(-1)
        indices = torch.arange(len(z), device=z.device)
        selected = margins[indices, weakest]
        gradient = torch.autograd.grad(selected.sum(), z, create_graph=False)[0]
        norm = gradient.norm(dim=-1)
        radius = selected / (scale * norm.clamp_min(1e-30))
        # A finite cap handles a locally flat decoder; its count is reported.
        radius = radius.clamp(-1e12, 1e12)
        clean = ((logits.argmax(-1) == targets) | ~valid).all(-1)
        return [{'clean_exact': bool(ok), 'weak_position': int(position),
                 'weak_margin': float(margin), 'gradient_norm_raw': float(g),
                 'local_radius_standardized_l2': float(r), 'flat_gradient': bool(g == 0)}
                for ok, position, margin, g, r in zip(clean.detach().cpu(), weakest.detach().cpu(),
                                                     selected.detach().cpu(), norm.detach().cpu(), radius.detach().cpu())]


def run(args):
    start = time.perf_counter()
    torch.set_num_threads(4)
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'candidate_risks.jsonl').exists():
        raise FileExistsError('Use a new output directory')
    inputs = read_rows(args.candidate_samples)
    rows = [row for row in inputs if row.get('fold') == args.fold and row['feasible']]
    available = len(rows)
    rows.sort(key=lambda row: hashlib.sha256((row['doc_id'] + ':' + row['unit_hex']).encode()).digest())
    rows = rows[:args.max_candidates]
    if not rows:
        raise ValueError('No clean feasible candidates in selected fold')
    with args.checkpoint.open('rb') as handle:
        checkpoint_sha = hashlib.sha256(handle.read()).hexdigest()
        handle.seek(0)
        saved = torch.load(handle, map_location='cpu', weights_only=False)
    manifest = args.candidate_samples.parent / 'manifest.json'
    if manifest.exists() and json.loads(manifest.read_text(encoding='utf-8'))['checkpoint_sha256'] != checkpoint_sha:
        raise ValueError('Candidate score checkpoint mismatch')
    ae = ByteStringAE(Cfg(saved['cfg'])).to(device=args.device, dtype=torch.float32).eval()
    ae.load_state_dict(saved['model'], strict=True)
    ae.requires_grad_(False)
    reference_rows = read_rows(args.reference_units)
    reference = [bytes.fromhex(row.get('hex', row.get('unit_hex'))) for row in reference_rows]
    strings = [bytes.fromhex(row['unit_hex']) for row in rows]
    unique = list(dict.fromkeys(reference + strings))
    vectors = encode(ae, unique, args.device, args.batch_size)
    lookup = {text: i for i, text in enumerate(unique)}
    rz = vectors[[lookup[text] for text in reference]].astype(np.float64)
    mean = rz.mean(0)
    scale = float(np.sqrt(np.mean((rz - mean) ** 2)))
    if not scale > 0:
        raise ValueError('Reference scale must be positive')
    z = vectors[[lookup[text] for text in strings]]
    clean_mismatches = []
    for offset in range(0, len(rows), args.batch_size):
        part = strings[offset:offset+args.batch_size]
        ids, lengths, _ = tensorize(part, ae.l_max, args.device)
        targets = targets_from_lengths(ids, lengths)
        batch_z = torch.tensor(z[offset:offset+len(part)], device=args.device)
        details = weakest_margin_gradient(ae, batch_z, targets, scale)
        for row, detail in zip(rows[offset:offset+len(part)], details):
            row['local_margin'] = detail
            if not detail['clean_exact']:
                clean_mismatches.append((row['doc_id'], row['unit_hex']))
    if clean_mismatches:
        raise RuntimeError(f'Clean candidate reconstruction mismatch: {clean_mismatches[:3]}')
    for row in rows:
        nll, entropy, width = row['target_nll_nats'], row['entropy_nats'], row['bytes']
        row['risk_scores'] = {'byte_length': float(width), 'total_nll': nll,
                              'mean_nll': nll / (width + 1), 'total_entropy': entropy,
                              'mean_entropy': entropy / (width + 1),
                              'negative_min_logit_margin': -row['min_logit_margin'],
                              'byte_risk': width * -math.expm1(-nll),
                              'negative_local_linear_radius': -row['local_margin']['local_radius_standardized_l2']}
        row['failures'] = {str(sigma): {'count': 0, 'trials': len(args.noise_seeds), 'by_seed': {}} for sigma in args.sigmas}
    for seed in args.noise_seeds:
        epsilon = []
        for row in rows:
            key = hashlib.sha256(f'{seed}:{row["unit_hex"]}'.encode()).digest()
            rng = np.random.default_rng(int.from_bytes(key[:8], 'little'))
            epsilon.append(rng.standard_normal(z.shape[1]).astype(np.float32))
        epsilon = np.asarray(epsilon)
        for sigma in args.sigmas:
            noisy = z + scale * sigma * epsilon
            with torch.no_grad():
                for offset in range(0, len(rows), args.batch_size):
                    part = strings[offset:offset+args.batch_size]
                    ids, lengths, _ = tensorize(part, ae.l_max, args.device)
                    targets = targets_from_lengths(ids, lengths)
                    valid = targets != -100
                    logits = decoder_logits(ae, torch.tensor(noisy[offset:offset+len(part)], device=args.device))
                    if not torch.isfinite(logits).all():
                        raise FloatingPointError('Nonfinite noisy decoder output')
                    exact = ((logits.argmax(-1) == targets) | ~valid).all(-1).cpu().tolist()
                    for row, ok in zip(rows[offset:offset+len(part)], exact):
                        row['failures'][str(sigma)]['count'] += int(not ok)
                        row['failures'][str(sigma)]['by_seed'][str(seed)] = int(not ok)
    assert all(param.grad is None and not param.requires_grad for param in ae.parameters())
    aucs = {}
    for sigma in args.sigmas:
        key = str(sigma)
        for row in rows:
            values = row['failures'][key]
            values['rate'] = values['count'] / values['trials']
        aucs[key] = {'failure_rate': sum(row['failures'][key]['count'] for row in rows) / (len(rows) * len(args.noise_seeds)),
                     'scores': {name: summarize_auc(rows, name, key) for name in rows[0]['risk_scores']}}
    with (output/'candidate_risks.jsonl').open('w', encoding='utf-8') as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False, separators=(',', ':'))+'\n')
    result = {'checkpoint_sha256': checkpoint_sha, 'checkpoint_step': saved.get('step'),
              'candidate_samples_sha256': sha(args.candidate_samples), 'reference_units_sha256': sha(args.reference_units),
              'script_sha256': sha(__file__), 'reference_scalar_scale': scale, 'reference_mean': mean.tolist(),
              'reference_count': len(reference), 'available_feasible_candidates': available,
              'candidate_count': len(rows), 'unique_strings': len(set(strings)),
              'flat_gradient_candidates': sum(row['local_margin']['flat_gradient'] for row in rows),
              'parameter_gradients_absent': True, 'protocol': {
                  'fold': args.fold, 'max_candidates': args.max_candidates, 'sigmas': args.sigmas, 'seeds': args.noise_seeds,
                  'noise_scale_rationale': args.noise_scale_rationale,
                  'precision': 'fp32', 'device': args.device, 'selection': 'clean feasible from AE-unfiltered hash candidates; bounded doc_id/unit hash order',
                  'noise': 'z+s_ref*sigma*epsilon; epsilon keyed by seed+unit bytes, shared with path diagnostics',
                  'failure': 'any target byte or true EOS argmax mismatch',
                  'auc': 'binary replicate labels with tied scores; higher score predicts failure; no fitted predictor',
                  'stratified_auc': 'sum concordant positive-negative trial pairs / sum positive-negative trial pairs within language x exact byte length',
                  'local_radius': 'weakest clean logit-margin position/runner-up only: m/(s_ref*||grad_z m||); standardized L2 distance, local linear proxy',
                  'gradients': 'only detached z requires grad inside local enable_grad; model weights frozen, no optimizer'},
              'auc': aucs, 'elapsed_seconds': time.perf_counter()-start}
    write_json(output/'results.json', result)
    print(json.dumps({'complete': True, 'candidates': len(rows), 'seconds': result['elapsed_seconds'],
                      'failure_rates': {key: val['failure_rate'] for key,val in aucs.items()}}), flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--candidate-samples', type=Path, required=True)
    p.add_argument('--reference-units', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--fold', default='analysis')
    p.add_argument('--max-candidates', type=int, default=4096)
    p.add_argument('--batch-size', type=int, default=256)
    p.add_argument('--device', default='cuda')
    p.add_argument('--sigmas', type=float, nargs='+', default=[.1,.2])
    p.add_argument('--noise-seeds', type=int, nargs='+', default=[42,123])
    p.add_argument('--noise-scale-rationale', default='User-specified noise levels')
    args=p.parse_args()
    if args.max_candidates<1 or any(sigma<=0 for sigma in args.sigmas):
        raise ValueError('Need positive candidate cap and noise scales')
    run(args)


if __name__=='__main__':main()
