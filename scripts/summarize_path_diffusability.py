"""Summarize frozen path diagnostics; paired document uncertainty and figures."""
from __future__ import annotations

import argparse
from collections import defaultdict
import gzip
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def common_strata(a, b):
    """Direct standardization to min estimated byte mass in shared lang/length cells."""
    sa, sb = a['by_language_and_length'], b['by_language_and_length']
    cells, numerator_a, numerator_b, mass = [], 0., 0., 0.
    for key in sorted(sa.keys() & sb.keys()):
        length = int(key.rsplit(':', 1)[1])
        wa = sa[key]['estimated_occurrence_weight'] * length
        wb = sb[key]['estimated_occurrence_weight'] * length
        weight = min(wa, wb)
        numerator_a += weight * sa[key]['em_byte_weighted']
        numerator_b += weight * sb[key]['em_byte_weighted']
        mass += weight
        cells.append(key)
    def total(x):
        return sum(v['estimated_occurrence_weight'] * int(k.rsplit(':', 1)[1]) for k, v in x.items())
    return {'cells': len(cells), 'common_mass': mass,
            'fraction_a': mass / total(sa), 'fraction_b': mass / total(sb),
            'em_a': numerator_a / mass, 'em_b': numerator_b / mass,
            'difference_pp_b_minus_a': 100 * (numerator_b - numerator_a) / mass}


def by_document(path, sigmas):
    groups = {str(s): defaultdict(lambda: np.zeros(2)) for s in sigmas}
    languages = {}
    with gzip.open(path, 'rt', encoding='utf-8') as handle:
        for line in handle:
            row = json.loads(line)
            sigma = str(row['sigma'])
            if sigma not in groups:
                continue
            doc = row['doc_id']
            languages[doc] = row['language']
            weight = row['weight'] * row['bytes']
            groups[sigma][doc] += [weight * row['exact'], weight]
    return groups, languages


def paired_bootstrap(a, b, languages, repeats=2000):
    docs = sorted(a.keys() & b.keys())
    aa, bb = np.array([a[d] for d in docs]), np.array([b[d] for d in docs])
    indices = defaultdict(list)
    for i, doc in enumerate(docs):
        indices[languages[doc]].append(i)
    rng = np.random.default_rng(20261104)
    draws = np.concatenate([rng.choice(ids, (repeats, len(ids)), replace=True) for ids in indices.values()], axis=1)
    va, vb = aa[draws].sum(1), bb[draws].sum(1)
    delta = 100 * (vb[:, 0] / vb[:, 1] - va[:, 0] / va[:, 1])
    return {'documents': len(docs), 'repeats': repeats,
            'difference_pp_b_minus_a': float(100 * (bb[:, 0].sum() / bb[:, 1].sum() - aa[:, 0].sum() / aa[:, 1].sum())),
            'percentile_95_pp': np.quantile(delta, [.025, .975]).tolist()}


LABELS = {'min_k': 'Min units', 'nll_0.1': 'NLL 0.1', 'nll_1': 'NLL 1', 'nll_10': 'NLL 10',
          'entropy_1': 'Entropy 1', 'byte_risk_1': 'Byte risk 1', 'margin_risk_1': 'Margin risk 1',
          'nll_1_regex_0.5': 'NLL + regex', 'nll_1_bpe_0.5': 'NLL + BPE',
          'nll_1_regex_0.5_samek': 'Regex, same K', 'nll_1_bpe_0.5_samek': 'BPE, same K',
          'random_samek_nll_1': 'Random, same K'}


def figures(core, result, out):
    out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.size': 10, 'font.style': 'normal', 'font.family': 'DejaVu Sans', 'figure.dpi': 150,
                         'axes.spines.top': False, 'axes.spines.right': False})
    exps = core['experiments']
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.7), constrained_layout=True)
    for ax, tag in zip(axes, ['4g', '8g']):
        for name, e in exps.items():
            if e['model_tag'] != tag or e['path_source_tag'] != tag:
                continue
            v = e['variant']
            if 'samek' in v or 'regex' in v or 'bpe' in v:
                continue
            x, y = e['bytes'] / e['units'], 100 * e['noise']['0.05']['overall']['em_byte_weighted']
            color = '#aa6688' if 'samek' in v else '#167b98'
            ax.scatter(x, y, c=color, marker='s' if 'samek' in v else 'o', s=35)
            ax.annotate(LABELS[v], (x, y), xytext=(3, -12) if v == 'margin_risk_1' else (3, 5), textcoords='offset points', fontsize=8)
        ax.set(title=f'{tag.upper()} AE: compression vs noise tolerance', xlabel='Bytes per unit (higher = fewer units)', ylabel='Byte-weighted exact match, sigma=0.05 (%)')
        ax.set_xlim(15.5, 32.5); ax.set_ylim(0, 90)
        ax.grid(alpha=.2)
    fig.savefig(out / 'compression_noise.png'); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for v, color in [('min_k', '#dc7534'), ('nll_1', '#2376ab'), ('nll_10', '#2b995c')]:
        for tag, style in [('4g', '--'), ('8g', '-')]:
            e = exps[f'{tag}__from_4g__{v}']
            axes[0].plot([float(s) for s in e['noise']], [100 * r['overall']['em_byte_weighted'] for r in e['noise'].values()], style, c=color, marker='o', ms=3, label=f'{LABELS[v]}, {tag}')
            axes[1].plot([int(d) for d in e['pca_projection']], [100 * r['overall']['em_byte_weighted'] for r in e['pca_projection'].values()], style, c=color, marker='o', ms=3, label=f'{LABELS[v]}, {tag}')
    axes[0].set(xlabel='Noise sigma / shared reference scale', ylabel='Byte-weighted exact match (%)', title='Same 4GiB boundaries, different AE checkpoints')
    axes[1].set(xlabel='Retained principal components', ylabel='Byte-weighted exact match (%)', title='Same boundaries: decoding after PCA projection')
    axes[0].legend(fontsize=7); axes[1].legend(fontsize=7)
    for ax in axes: ax.grid(alpha=.2)
    fig.savefig(out / 'training_fixed_paths.png'); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for ax, tag in zip(axes, ['4g', '8g']):
        for v in ['nll_1', 'nll_1_regex_0.5_samek', 'nll_1_bpe_0.5_samek', 'random_samek_nll_1']:
            e = exps[f'{tag}__from_{tag}__{v}']
            ax.plot([float(s) for s in e['noise']], [100 * r['overall']['em_byte_weighted'] for r in e['noise'].values()], marker='o', ms=3, label=LABELS[v])
        ax.set(title=f'{tag.upper()}: identical unit counts per document', xlabel='Noise sigma / reference scale', ylabel='Byte-weighted exact match (%)')
        ax.grid(alpha=.2); ax.legend(fontsize=8)
    fig.savefig(out / 'same_k_priors.png'); plt.close(fig)

    risk_path = out.parent / 'risk_8g/results.json'
    if risk_path.exists():
        risk = read(risk_path)['auc']
        keys = ['byte_length', 'total_nll', 'total_entropy', 'negative_min_logit_margin', 'negative_local_linear_radius']
        labels = ['Length', 'NLL', 'Entropy', 'Min margin', 'Local radius']
        fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
        x = np.arange(len(keys))
        for shift, sigma, color in [(-.18, '0.025', '#2376ab'), (.18, '0.05', '#dc7534')]:
            ys = [risk[sigma]['scores'][k]['same_language_and_exact_byte_length']['auc'] for k in keys]
            bars = ax.bar(x + shift, ys, .34, label=f'sigma={sigma}', color=color)
            ax.bar_label(bars, fmt='%.3f', fontsize=8, padding=3)
        ax.axhline(.5, color='gray', ls='--', lw=1)
        ax.set(xticks=x, xticklabels=labels, ylim=(0, .9), ylabel='AUROC within language x exact byte length', title='8GiB AE: predicting noise failures from clean candidates')
        ax.legend(); fig.savefig(out / 'candidate_risk.png'); plt.close(fig)

    book_path = out.parent / 'codebooks_readout/results.json'
    if book_path.exists():
        books = read(book_path)['experiments']
        fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
        for ax, tag in zip(axes, ['4g', '8g']):
            for variant, color in [('nll_1', '#2376ab'), ('nll_10', '#dc7534')]:
                vals = books[f'{tag}__{variant}']['times']
                for key, style, label in [('posterior_mean_decoder', '-', 'mean -> decoder'), ('map_native_bytes', '--', 'MAP bytes')]:
                    ax.plot([float(t) for t in vals], [100 * r[key]['em_unit'] for r in vals.values()], style, color=color, marker='o', ms=3, label=f'{LABELS[variant]}, {label}')
            ax.set(title=f'{tag.upper()}: analytic closed-codebook readout', xlabel='Flow noise time t', ylabel='Exact identity / decoded unit (%)', ylim=(-3, 103))
            ax.legend(fontsize=7); ax.grid(alpha=.2)
        fig.savefig(out / 'codebook_readout.png'); plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=Path('runs/path_diffusability_20261104'))
    a = p.parse_args()
    core = read(a.root / 'latents/results.json')
    exps = core['experiments']
    directories = dict.fromkeys(exps, a.root / 'latents')
    extra_file = a.root / 'latents_low_lambda/results.json'
    if extra_file.exists():
        extra = read(extra_file)
        exps.update(extra['experiments'])
        directories.update(dict.fromkeys(extra['experiments'], extra_file.parent))
    rows = []
    for name, e in exps.items():
        g = e['geometry']
        rows.append({'name': name, 'model': e['model_tag'], 'source': e['path_source_tag'], 'variant': e['variant'],
                     'bytes_per_unit': e['bytes'] / e['units'], 'units': e['units'], 'bytes': e['bytes'],
                     'pr': g['participation_ratio'], 'effective_rank': g['entropy_effective_rank'],
                     'twonn': g['local']['twonn_log_ratio_mle'], 'own_to_reference_scale': e['own_to_reference_scale'],
                     'em_byte': {s: r['overall']['em_byte_weighted'] for s, r in e['noise'].items()},
                     'pca_em_byte': {s: r['overall']['em_byte_weighted'] for s, r in e['pca_projection'].items()}})
    pairs = []
    for tag in ['4g', '8g']:
        for va, vb in [('min_k', 'nll_0.1'), ('min_k', 'nll_1'), ('nll_1', 'nll_10'), ('nll_1', 'nll_1_regex_0.5_samek'),
                       ('nll_1', 'nll_1_bpe_0.5_samek'), ('random_samek_nll_1', 'nll_1')]:
            pairs.append((f'{tag}__from_{tag}__{va}', f'{tag}__from_{tag}__{vb}'))
    for v in ['min_k', 'nll_1', 'nll_10']:
        pairs.append((f'4g__from_4g__{v}', f'8g__from_4g__{v}'))
        pairs.append((f'8g__from_4g__{v}', f'8g__from_8g__{v}'))
    cached, comparisons = {}, []
    for na, nb in pairs:
        for n in [na, nb]:
            if n not in cached:
                cached[n] = by_document(directories[n] / f'noise_{n}.jsonl.gz', [.025, .05])
        comp = {'a': na, 'b': nb, 'noise': {}}
        for sigma in ['0.025', '0.05']:
            comp['noise'][sigma] = {
                'paired_document_bootstrap': paired_bootstrap(cached[na][0][sigma], cached[nb][0][sigma], cached[na][1]),
                'common_language_length': common_strata(exps[na]['noise'][sigma], exps[nb]['noise'][sigma])}
        comparisons.append(comp)
    result = {'protocol': {'bootstrap': '2000 paired document resamples within each language; conditional on the measured unit/noise panel',
                           'common_strata': 'overlap language x exact byte length; min estimated byte mass, renormalized; not paired strings',
                           'scope': 'analysis documents only; no model training'}, 'table': rows, 'comparisons': comparisons}
    (a.root / 'analysis_summary.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    figures(core, result, a.root / 'figures')
    print(json.dumps({'groups': len(rows), 'comparisons': len(comparisons), 'output': str(a.root / 'analysis_summary.json')}))


if __name__ == '__main__':
    main()
