"""Paired summaries and figures for the frozen covariance-noise ablation."""
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

from summarize_path_diffusability import paired_bootstrap


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def read_panel(path):
    by_sigma = defaultdict(list)
    with gzip.open(path, 'rt', encoding='utf-8') as handle:
        for line in handle:
            row = json.loads(line)
            by_sigma[str(row['sigma'])].append(row)
    return by_sigma


def key(row):
    return row['doc_id'], row['position'], row['hex'], row['seed']


def paired_rows(a, b):
    aa = {key(r): r for r in a}
    bb = {key(r): r for r in b}
    assert len(aa) == len(a) == len(b) == len(bb) and aa.keys() == bb.keys(), 'Different paired panels'
    da, db = defaultdict(lambda: np.zeros(2)), defaultdict(lambda: np.zeros(2))
    languages, flips = {}, defaultdict(float)
    energy_a, energy_b, mass = 0., 0., 0.
    for k, ra in aa.items():
        rb = bb[k]
        assert ra['weight'] == rb['weight'] and ra['bytes'] == rb['bytes']
        w = ra['weight'] * ra['bytes']
        doc = ra['doc_id']
        languages[doc] = ra['language']
        da[doc] += [w * ra['exact'], w]
        db[doc] += [w * rb['exact'], w]
        flips[f'{int(ra["exact"])}_to_{int(rb["exact"])}'] += w
        energy_a += w * ra['actual_noise_mse']
        energy_b += w * rb['actual_noise_mse']
        mass += w
    return {'paired_document_bootstrap': paired_bootstrap(da, db, languages),
            'weighted_byte_flip_fractions': {k: v / mass for k, v in flips.items()},
            'byte_weighted_noise_mse_a': energy_a / mass,
            'byte_weighted_noise_mse_b': energy_b / mass,
            'actual_energy_ratio_b_over_a': energy_b / energy_a,
            'sample_noise_pairs': len(a)}


def make_figures(result, out):
    out.mkdir(parents=True, exist_ok=True)
    plt.rcdefaults()
    plt.rcParams.update({'font.family': 'Arial', 'font.size': 10, 'figure.dpi': 150,
                         'axes.spines.top': False, 'axes.spines.right': False})
    rows = {(r['group'], r['sigma']): r for r in result['groups']}
    colors = {'isotropic': '#6f7884', 'aligned': '#168e9b', 'rotated': '#d3853d'}
    labels = {'isotropic': 'Isotropic', 'aligned': 'Covariance aligned', 'rotated': 'Same-energy rotation'}
    variants = ['min_k', 'nll_0.1', 'nll_1', 'nll_10', 'byte_risk_1']
    ticklabels = ['Min units', 'NLL 0.1', 'NLL 1', 'NLL 10', 'Byte risk 1']
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True, sharey=True)
    x = np.arange(len(variants))
    for i, sigma in enumerate(['0.025', '0.05']):
        for j, tag in enumerate(['4g', '8g']):
            ax = axes[i, j]
            for k, arm in enumerate(['isotropic', 'aligned', 'rotated']):
                y = [100 * rows[(f'{tag}__from_{tag}__{v}', sigma)]['arms'][arm]['em_byte_weighted'] for v in variants]
                bars = ax.bar(x + (k - 1) * .24, y, .23, color=colors[arm], label=labels[arm])
                ax.bar_label(bars, fmt='%.1f', fontsize=7, padding=2)
            ax.set(title=f'{tag.upper()} AE, sigma={sigma}', xticks=x, xticklabels=ticklabels,
                   ylabel='Byte-weighted exact match (%)', ylim=(0, 110))
            ax.grid(axis='y', alpha=.15); ax.set_axisbelow(True)
    fig.legend(*axes[0, 0].get_legend_handles_labels(), fontsize=9, loc='upper center',
               bbox_to_anchor=(.5, 1.03), ncol=3)
    fig.savefig(out / 'direction_comparison.png', bbox_inches='tight'); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    variants = ['nll_1', 'nll_1_regex_0.5_samek', 'nll_1_bpe_0.5_samek', 'random_samek_nll_1']
    ticklabels = ['NLL', 'Regex, same K', 'BPE, same K', 'Random, same K']
    x = np.arange(len(variants))
    for ax, tag in zip(axes, ['4g', '8g']):
        for arm in ['isotropic', 'aligned', 'rotated']:
            y = [100 * rows[(f'{tag}__from_{tag}__{v}', '0.05')]['arms'][arm]['em_byte_weighted'] for v in variants]
            ax.plot(x, y, marker='o', ms=4, c=colors[arm], label=labels[arm])
        ax.set(title=f'{tag.upper()}: fixed unit counts, sigma=0.05', xticks=x, xticklabels=ticklabels,
               ylabel='Byte-weighted exact match (%)', ylim=(0, 105))
        ax.grid(alpha=.15); ax.legend(fontsize=8)
    fig.savefig(out / 'same_k_directions.png'); plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=Path('runs/path_diffusability_20261104/covariance_noise'))
    args = p.parse_args()
    source = read_json(args.root / 'results.json')
    result = {'protocol': {'uncertainty': '2000 paired document resamples within language, conditional on fixed units/noise seeds/Q',
                           'energy': 'byte-weighted mean per-coordinate squared standardized perturbation; ratio uses realized samples',
                           'isotropic': 'existing readout reused; same Gaussian seeds as aligned but energy matched in expectation',
                           'rotated': 'orthogonal rotation of aligned perturbation; pairwise norm preserved before FP32 rounding'},
              'groups': [], 'models': source['models']}
    for group, exp in source['experiments'].items():
        panels = {arm: read_panel(args.root / f'noise_{group}__{arm}.jsonl.gz') for arm in ['isotropic', 'aligned', 'rotated']}
        for sigma in ['0.025', '0.05']:
            row = {'group': group, 'model_tag': exp['model_tag'], 'path_source_tag': exp['path_source_tag'],
                   'variant': exp['variant'], 'sigma': sigma, 'arms': {}, 'comparisons': {}}
            for arm in panels:
                row['arms'][arm] = exp['arms'][arm]['noise'][sigma]['overall']
            for a, b in [('isotropic', 'aligned'), ('aligned', 'rotated'), ('isotropic', 'rotated')]:
                row['comparisons'][f'{a}_to_{b}'] = paired_rows(panels[a][sigma], panels[b][sigma])
            result['groups'].append(row)
    (args.root / 'summary.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    make_figures(result, args.root / 'figures')
    print(json.dumps({'groups_by_sigma': len(result['groups']), 'out': str(args.root / 'summary.json')}))


if __name__ == '__main__':
    main()
