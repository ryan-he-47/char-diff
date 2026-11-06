"""Plot two saved, length-stratified AE evaluations without model inference."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--before', type=Path, required=True)
    p.add_argument('--after', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--stage-dir', type=Path)
    args = p.parse_args()
    before, after = [json.loads(path.read_text(encoding='utf-8')) for path in [args.before, args.after]]
    plt.rcParams.update({'font.family': 'Microsoft YaHei', 'font.size': 10,
                         'axes.unicode_minus': False, 'axes.spines.top': False,
                         'axes.spines.right': False, 'savefig.facecolor': 'white'})
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.6), sharex=True, sharey=True, layout='constrained')
    for ax, key, title in zip(axes.flat, [None, 'chinese', 'english', 'korean'],
                              ['全部12语（每语种、每长度256条）', '中文', '英文', '韩文']):
        for data, color in [(before, '#9a5a4b'), (after, '#167c9d')]:
            nat = data['natural_stratified']
            values = nat['by_byte_length'] if key is None else nat['by_language_and_length'][key]
            length = sorted(map(int, values))
            gib = round(data['training_state']['decay_start_bytes'] / 1024**3)
            ax.plot(length, [values[str(n)]['em'] * 100 for n in length],
                    color=color, lw=2, marker='.', ms=4, label=f'{gib} GiB + 64 MiB 退火')
        ax.set_title(title)
        ax.set_ylim(0, 102)
        ax.set_xlim(.5, 32.5)
        ax.set_xticks([1, 4, 8, 12, 16, 20, 24, 28, 32])
        ax.grid(alpha=.18)
        ax.legend(frameon=False, loc='lower left')
    for ax in axes[1]:
        ax.set_xlabel('片段字节长度')
    for ax in axes[:, 0]:
        ax.set_ylabel('整串重建 EM（%，含 EOS）')
    fig.suptitle('继续训练后的长度曲线 · 完全相同的98,304个自然片段 · FP32', fontsize=14)
    for ext in ['png', 'svg']:
        fig.savefig(args.output_dir / f'length_progress.{ext}', dpi=180)
    plt.close(fig)

    if args.stage_dir:
        stages = {phase: [json.loads(path.read_text(encoding='utf-8'))
                          for path in sorted(args.stage_dir.glob(f'{phase}_*.json'))]
                  for phase in ['stable', 'cooldown']}
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.3), layout='constrained')
        for ax, panel, title in zip(axes, ['natural', 'augmented'], ['自然片段', '配对增强片段']):
            for phase, color, label in [('stable', '#9b9b9b', 'Stable 检查点'),
                                         ('cooldown', '#167c9d', '同阶段 +64 MiB 退火')]:
                rows = stages[phase]
                xs = [(row['training_state'].get('decay_start_bytes') or row['training_state']['seen_bytes'])
                      / 1024**3 for row in rows]
                ys = [row['panels'][panel]['overall']['string_em'] * 100 for row in rows]
                ax.plot(xs, ys, marker='o', ms=4, color=color, label=label)
            ax.set_title(title)
            ax.set_xlabel('Stable 累计训练量（GiB目标字节）')
            ax.set_ylabel('整串重建 EM（%）')
            ax.set_ylim(0, 102)
            ax.grid(alpha=.18)
            ax.legend(frameon=False, loc='lower right')
        fig.suptitle('WSD阶段评估 · 固定2048自然 + 2048配对增强片段', fontsize=13)
        for ext in ['png', 'svg']:
            fig.savefig(args.output_dir / f'wsd_progress.{ext}', dpi=180)
        plt.close(fig)
    print(args.output_dir)


if __name__ == '__main__':
    main()
