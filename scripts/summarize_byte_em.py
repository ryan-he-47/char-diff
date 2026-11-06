"""Summarize completed byte-EM outputs and paired window uncertainty."""
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from char_diff.utils import ROOT, write_json, sha256

LABELS = {'x0_euler': 'x₀控制 / Euler', 'gmm_early_euler': 'GMM early / Euler',
          'gmm_early_gm_ode': 'GMM early / GM-ODE', 'gmm_late_euler': 'GMM late / Euler',
          'gmm_late_gm_ode': 'GMM late / GM-ODE', 'ar_sample': '公开AR / T=1采样',
          'ar_greedy': '公开AR / 贪心'}


def main():
    folder = ROOT / 'runs/cross_model_byte_em_20261102'
    docs = ROOT / 'docs/experiments/cross-model-byte-em'
    summary = json.loads((folder / 'summary.json').read_text(encoding='utf-8'))
    data = json.loads((folder / 'anchors.json').read_text(encoding='utf-8'))
    cases = data['cases']; windows = data['windows']; windex = {w: i for i, w in enumerate(windows)}
    selection = {'anchors': len(cases), 'unique_doc_positions': len({(c['doc'], c['anchor']) for c in cases}),
                 'target_bytes': sum(len(bytes.fromhex(c['target_hex'])) for c in cases),
                 'history_bpe_min_median_max': np.quantile([len(c['input_ids']) for c in cases], [0, .5, 1]).tolist(),
                 'empty_history': sum(not c['input_ids'] for c in cases),
                 'horizon4_eligible': sum(len(bytes.fromhex(c['reference_hex'])) >= 4 for c in cases),
                 'horizon8_eligible': sum(len(bytes.fromhex(c['reference_hex'])) >= 8 for c in cases)}
    raw = {}; counts = {}; example_rows = {}
    for label in LABELS:
        raw[label] = [json.loads(line) for line in (folder / f'{label}.jsonl').read_text(encoding='utf-8').splitlines()]
        seeds = [0] if label == 'ar_greedy' else [42, 123]
        assert {(r['id'], r['seed']) for r in raw[label]} == {(c['id'], s) for c in cases for s in seeds}
        assert len(raw[label]) == len(cases) * len(seeds)
        counts[label] = np.zeros((len(windows), 2))
        for row in raw[label]:
            counts[label][windex[row['window']]] += [row['exact_bytes'], row['target_bytes']]
        example_rows[label] = {r['id']: r for r in raw[label] if r['seed'] == seeds[0]}
    draws = np.random.default_rng(20261102).integers(0, len(windows), (5000, len(windows)))
    def rate(label):
        total = counts[label][draws].sum(1)
        return total[:, 0] / total[:, 1]
    control = rate('x0_euler')
    uncertainty = {'unit': 'percentage points', 'cluster': 'selected dev window, paired across models',
                   'draws': 5000, 'seed': 20261102, 'delta_vs_x0': {}}
    for label in LABELS:
        if label == 'x0_euler': continue
        delta = 100 * (rate(label) - control)
        observed = 100 * (summary['models'][label]['metrics']['byte_weighted_em'] -
                          summary['models']['x0_euler']['metrics']['byte_weighted_em'])
        uncertainty['delta_vs_x0'][label] = {'observed_pp': observed,
                                            'paired_window_bootstrap_95_percent': np.quantile(delta, [.025, .975]).tolist()}
    horizon_rates = {}
    for label in ['x0_euler', 'ar_sample']:
        values = np.zeros((len(windows), 2))
        for row in raw[label]:
            if row['em8'] is not None: values[windex[row['window']]] += [row['em8'], 1]
        totals = values[draws].sum(1)
        horizon_rates[label] = totals[:, 0] / totals[:, 1]
    uncertainty['ar_sample_minus_x0_em8'] = {
        'observed_pp': 100 * (summary['models']['ar_sample']['metrics']['em8'] - summary['models']['x0_euler']['metrics']['em8']),
        'paired_window_bootstrap_95_percent': np.quantile(100 * (horizon_rates['ar_sample'] - horizon_rates['x0_euler']), [.025, .975]).tolist()}
    full_ids = {c['id'] for c in cases if c['block_length'] == 16}
    full_block = {'anchors': len(full_ids), 'models': {}}
    for label in LABELS:
        rows = [r for r in raw[label] if r['id'] in full_ids]
        full_block['models'][label] = {'span_em': sum(r['exact'] for r in rows) / len(rows),
                                      'byte_weighted_em': sum(r['exact_bytes'] for r in rows) / sum(r['target_bytes'] for r in rows)}
    write_json(folder / 'paired-window-bootstrap.json', uncertainty)
    evidence = docs / 'evidence'; evidence.mkdir(parents=True, exist_ok=True)
    write_json(evidence / 'summary.json', summary)
    write_json(evidence / 'paired-window-bootstrap.json', uncertainty)
    write_json(evidence / 'targeted-checks.json', json.loads((folder / 'targeted-checks.json').read_text(encoding='utf-8')))
    write_json(evidence / 'selection.json', selection)
    write_json(evidence / 'full-block-subset.json', full_block)
    write_json(evidence / 'source-hashes.json', {name: sha256(ROOT / name) for name in
               ['scripts/compare_byte_em.py', 'scripts/check_byte_em.py', 'scripts/review_gmm.py',
                'char_diff/model.py', 'char_diff/gmm.py', 'char_diff/codec.py',
                'runs/continuation_review_20261101/uniform/last.pt']})
    pct = lambda v: f'{100*v:.2f}%' if v is not None else '—'
    lines = ['| 模型与采样 | 第一unit跨度EM | 字节加权EM | 固定4字节EM | 固定8字节EM |',
             '|---|---:|---:|---:|---:|']
    for label, name in LABELS.items():
        m = summary['models'][label]['metrics']
        lines.append('| ' + name + ' | ' + ' | '.join(pct(m[k]) for k in ['span_em', 'byte_weighted_em', 'em4', 'em8']) + ' |')
    lines += ['', '## 真实unit字节长度分桶：普通跨度EM', '',
              '| 模型 | 1 | 2 | 3–4 | 5–8 | 9–16 | 17–32 |', '|---|---:|---:|---:|---:|---:|---:|']
    for label, name in LABELS.items():
        bins = summary['models'][label]['metrics']['by_length']
        lines.append('| ' + name + ' | ' + ' | '.join(pct(v['span_em']) for v in bins.values()) + ' |')
    lines += ['', '每种采样模型各桶预测次数：' + ', '.join(f"{k}: {v['cases']}" for k,v in summary['models']['x0_euler']['metrics']['by_length'].items()) + '。贪心次数为对应一半。',
              '', '## 两种种子：字节加权EM', '', '| 模型 | seed42 | seed123 |', '|---|---:|---:|']
    for label, name in LABELS.items():
        if label == 'ar_greedy': continue
        seeds = summary['models'][label]['metrics']['by_seed']
        lines.append(f"| {name} | {pct(seeds['42']['byte_weighted_em'])} | {pct(seeds['123']['byte_weighted_em'])} |")
    lines += ['', '## 相对x₀的字节加权EM差值', '',
              '按64个窗口成组、模型配对重采样5,000次，保留同一窗口内的位置与两个噪声种子。以下区间描述这组dev窗口的抽样波动。', '',
              '| 模型 | 差值（百分点） | 配对窗口bootstrap 95%区间 |', '|---|---:|---:|']
    for label, value in uncertainty['delta_vs_x0'].items():
        lo, hi = value['paired_window_bootstrap_95_percent']
        lines.append(f"| {LABELS[label]} | {value['observed_pp']:+.2f} | [{lo:+.2f}, {hi:+.2f}] |")
    (docs / 'tables.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    # Fixed random case selection, independent of wins/losses.
    selected = sorted(np.random.default_rng(20261102).choice(len(cases), 12, replace=False).tolist())
    samples = ['# 共同前文的12个固定示例', '', 'seed20261102随机抽取锚点，采样模型展示seed42，贪心展示唯一输出。生成片段最多32字节；评分以真实首unit的长度为准。', '']
    for i in selected:
        c = cases[i]
        decode = lambda h: bytes.fromhex(h).decode('utf-8', errors='replace')
        samples += [f"## 锚点{i} / doc{c['doc']} / unit{c['anchor']}", '',
                    '前文末尾：', '', '```text', decode(c['history_hex'])[-240:], '```', '',
                    '真实首unit：' + repr(decode(c['target_hex'])), '', '真实后续片段：' + repr(decode(c['reference_hex'])), '']
        for label, name in LABELS.items():
            row = example_rows[label][i]
            samples += [f"- {name}，命中={row['exact']}：{repr(decode(row['predicted_hex']))}"]
        samples.append('')
    (docs / 'samples.md').write_text('\n'.join(samples) + '\n', encoding='utf-8')
    print(json.dumps({'metrics': {k:{m:v['metrics'][m] for m in ['span_em','byte_weighted_em','em4','em8']} for k,v in summary['models'].items()},
                      'uncertainty': uncertainty, 'elapsed_seconds': summary['elapsed_seconds']}, indent=2))


if __name__ == '__main__':
    main()
