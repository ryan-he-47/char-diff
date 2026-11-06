"""Render the saved length experiment; no model inference."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('results', type=Path)
    args = parser.parse_args()
    data = json.loads(args.results.read_text(encoding='utf-8'))
    out = args.results.parent
    plt.rcParams.update({'font.family': 'Microsoft YaHei', 'font.size': 10,
                         'axes.unicode_minus': False, 'axes.spines.top': False,
                         'axes.spines.right': False, 'savefig.facecolor': 'white'})
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.8), layout='constrained')
    natural = data['natural_stratified']
    curves = [('全部语言', natural['by_byte_length'], '#202c39', 2.5),
              ('英文', natural['by_language_and_length']['english'], '#1976d2', 1.5),
              ('中文', natural['by_language_and_length']['chinese'], '#c0392b', 1.5),
              ('韩文', natural['by_language_and_length']['korean'], '#be7900', 1.5)]
    for label, values, color, width in curves:
        lengths = sorted(map(int, values))
        axes[0].plot(lengths, [100 * values[str(n)]['em'] for n in lengths],
                     label=label, color=color, lw=width, marker='.', ms=4)
    axes[0].set_title('任意 UTF-8 片段：EM 随字节长度变化')
    axes[0].set_ylabel('整串重建 EM（%，含 EOS）')
    axes[0].legend(loc='lower left', frameon=False)
    paired = data['matched_all_legacy_units']['by_8byte_bucket']
    buckets = list(paired)
    for key, label, color in [('word', '旧整词片段', '#1976d2'),
                              ('arbitrary', '同文档、同长度任意片段', '#de6f21')]:
        axes[1].plot(range(4), [100 * paired[b][key]['em'] for b in buckets],
                     label=label, color=color, lw=2, marker='.', ms=5)
    axes[1].set_title('同文档、同字节长度配对（按区间汇总）')
    axes[1].legend(loc='lower left', frameon=False)
    for ax in axes:
        ax.set_xlabel('片段字节长度')
        ax.set_ylim(0, 102)
        ax.grid(alpha=.18)
    axes[0].set_xlim(.5, 32.5)
    axes[0].set_xticks([1, 4, 8, 12, 16, 20, 24, 28, 32])
    axes[1].set_xlim(-.15, 3.15)
    axes[1].set_xticks(range(4), [f'{b}\nn={paired[b]["n"]:,}' for b in buckets])
    axes[1].text(.98, .18, '25–32 B 配对中，中文占 97.6%', transform=axes[1].transAxes,
                 ha='right', fontsize=9, color='#555')
    fig.suptitle('4 GiB + 64 MiB 退火 AE · 固定权重 · 校准集 · FP32', fontsize=14)
    fig.savefig(out / 'em_by_length.png', dpi=180)
    fig.savefig(out / 'em_by_length.svg')
    plt.close(fig)

    paired_exact = data['matched_all_legacy_units']['by_byte_length']
    lengths = np.arange(1, 33)
    fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True,
                             gridspec_kw={'height_ratios': [3, 1]}, layout='constrained')
    for key, label, color in [('word', '旧整词片段', '#1976d2'),
                              ('arbitrary', '同文档同长度任意片段', '#de6f21')]:
        ys = [100 * paired_exact[str(n)][key]['em'] if str(n) in paired_exact else np.nan for n in lengths]
        axes[0].plot(lengths, ys, label=label, color=color, marker='.', lw=1.5)
    counts = [paired_exact.get(str(n), {}).get('n', 0) for n in lengths]
    axes[1].bar(lengths, counts, color='#718096')
    axes[1].set_yscale('symlog', linthresh=10)
    axes[1].set_ylabel('配对数')
    axes[1].set_xlabel('片段字节长度（29、32 B 无旧整词对照）')
    axes[1].set_xticks(lengths)
    axes[0].set_ylim(0, 102)
    axes[0].set_ylabel('整串 EM（%）')
    axes[0].legend(frameon=False)
    axes[0].set_title('精确长度对照及样本数：长片段的语言构成随可用整词变化')
    for ax in axes:
        ax.grid(alpha=.18)
    fig.savefig(out / 'matched_exact_lengths.png', dpi=180)
    fig.savefig(out / 'matched_exact_lengths.svg')
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.8), layout='constrained')
    for ax, key, title in [(axes[0], 'by_8byte_bucket', '按长度区间'),
                            (axes[1], 'by_language', '按语言')]:
        values = data['matched_all_legacy_units'][key]
        labels = list(values)
        delta = np.array([values[k]['word_minus_arbitrary_em'] * 100 for k in labels])
        intervals = np.array([values[k]['document_bootstrap_delta95'] for k in labels]) * 100
        y = np.arange(len(labels))
        ax.barh(y, delta, color=['#277e64' if x >= 0 else '#9a5cad' for x in delta], alpha=.8)
        ax.hlines(y, intervals[:, 0], intervals[:, 1], color='#222', linewidth=1.4)
        ax.plot(delta, y, 'o', ms=3, color='#222')
        ax.axvline(0, color='#444', linewidth=.8)
        ax.set_yticks(y, labels)
        ax.invert_yaxis()
        ax.set_xlabel('整词 EM − 同长度任意片段 EM（百分点）')
        ax.set_title(title)
        ax.grid(axis='x', alpha=.18)
    fig.suptitle('同文档配对差值 · 95% 文档聚类 bootstrap 区间（300 次）', fontsize=13)
    fig.savefig(out / 'matched_difference.png', dpi=180)
    fig.savefig(out / 'matched_difference.svg')
    plt.close(fig)
    print(out / 'em_by_length.png')
    print(out / 'matched_difference.png')


if __name__ == '__main__':
    main()
