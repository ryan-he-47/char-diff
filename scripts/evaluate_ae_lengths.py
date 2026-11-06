"""Length-stratified UTF-8 AE reconstruction and matched old-word comparison."""
from __future__ import annotations

import argparse
from array import array
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import random
import sys
import time
import unicodedata

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch

from char_diff.ae_quality import AEQualityScorer
from char_diff.utils import write_json
from char_diff.vendor.bltz.config import Cfg
from char_diff.vendor.bltz.models.autoencoder import ByteStringAE
from char_diff.vendor.bltz.segment_v2 import segment_word


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def span_index(text, max_bytes=32):
    offsets = [0]
    for c in text:
        offsets.append(offsets[-1] + len(c.encode('utf-8')))
    starts = [array('I') for _ in range(max_bytes + 1)]
    for i, start in enumerate(offsets[:-1]):
        for end in offsets[i + 1:i + 1 + max_bytes]:
            length = end - start
            if length > max_bytes:
                break
            starts[length].append(start)
    return offsets, starts


def build_samples(records, *, per_cell=256, seed=1234):
    """Same-document, same-byte-length pairs; no AE-dependent selection."""
    rng = random.Random(seed)
    docs, supports, reservoirs, counts = [], defaultdict(list), {}, defaultdict(int)
    segmentation = defaultdict(lambda: defaultdict(int))
    for record in records:
        if record['split'] != 'calibration':
            continue
        text = record['text']
        raw = text.encode('utf-8', errors='strict')
        offsets, starts = span_index(text)
        index = len(docs)
        # Raising the cap returns the exact legacy pre-BPE word runs.
        original_units = segment_word(raw, '', l_max=max(32, len(raw)), merge='prev')
        assert b''.join(original_units) == raw
        doc = {'id': str(record['id']), 'language': record['language'], 'raw': raw,
               'offsets': offsets, 'starts': starts, 'word_spans': set()}
        docs.append(doc)
        lang = record['language']
        for length in range(1, 33):
            if starts[length]:
                supports[(lang, length)].append(index)
        pos = 0
        stats = segmentation[lang]
        stats['documents'] += 1
        stats['source_bytes'] += len(raw)
        for unit in original_units:
            length = len(unit)
            stats['raw_units'] += 1
            stats['raw_unit_bytes'] += length
            lexical = any(unicodedata.category(c).startswith('L') for c in unit.decode('utf-8'))
            stats['lexical_raw_units'] += int(lexical)
            if length > 32:
                stats['over32_units'] += 1
                stats['over32_bytes'] += length
                stats['lexical_over32_units'] += int(lexical)
            else:
                stats['eligible_units'] += 1
                stats['eligible_bytes'] += length
                stats['lexical_eligible_units'] += int(lexical)
                doc['word_spans'].add((pos, pos + length))
                cell = (lang, length)
                counts[cell] += 1
                item = (index, pos, lexical)
                reservoir = reservoirs.setdefault(cell, [])
                if len(reservoir) < per_cell:
                    reservoir.append(item)
                else:
                    pick = rng.randrange(counts[cell])
                    if pick < per_cell:
                        reservoir[pick] = item
            pos += length
    for lang, stats in segmentation.items():
        for name in ['over32_units', 'over32_bytes', 'lexical_over32_units']:
            stats[name] += 0
        stats['eligible_unit_fraction'] = stats['eligible_units'] / stats['raw_units']
        stats['eligible_byte_fraction'] = stats['eligible_bytes'] / stats['source_bytes']

    def sample(doc_index, start, length):
        doc = docs[doc_index]
        raw = doc['raw'][start:start + length]
        assert len(raw) == length
        text = raw.decode('utf-8', errors='strict')
        return {'doc_id': doc['id'], 'language': doc['language'], 'start_byte': start,
                'end_byte': start + length, 'bytes': length, 'text': text,
                'characters': len(text), 'is_legacy_word_span': (start, start + length) in doc['word_spans'],
                'contains_letter': any(unicodedata.category(c).startswith('L') for c in text)}

    natural = []
    for cell, eligible_docs in sorted(supports.items()):
        lang, length = cell
        for _ in range(per_cell):
            index = rng.choice(eligible_docs)
            start = rng.choice(docs[index]['starts'][length])
            natural.append(sample(index, start, length))
    pairs = []
    for (lang, length), reservoir in sorted(reservoirs.items()):
        for index, start, _ in reservoir:
            arbitrary_start = rng.choice(docs[index]['starts'][length])
            pairs.append({'pair_id': len(pairs), 'language': lang, 'bytes': length,
                          'word': sample(index, start, length),
                          'arbitrary': sample(index, arbitrary_start, length)})
    support = [{'language': lang, 'bytes': length, 'eligible_documents': len(supports.get((lang, length), [])),
                'legacy_occurrences': counts[(lang, length)],
                'matched_pairs': len(reservoirs.get((lang, length), []))}
               for lang in sorted(segmentation) for length in range(1, 33)]
    return natural, pairs, dict(segmentation), support


def aggregate(rows):
    n = len(rows)
    if not n:
        return {'n': 0, 'em': None}
    exact = sum(r['exact'] for r in rows)
    p = exact / n
    z = 1.959963984540054
    den = 1 + z * z / n
    center = (p + z * z / (2 * n)) / den
    radius = z / den * (p * (1 - p) / n + z * z / (4 * n * n))**.5
    return {'n': n, 'exact': exact, 'em': p, 'wilson95': [center - radius, center + radius],
            'nll_per_byte_eos': sum(r['target_nll_nats'] for r in rows) / sum(r['bytes'] + 1 for r in rows),
            'unique_spans': len({(r['doc_id'], r['start_byte'], r['end_byte']) for r in rows}),
            'unique_strings': len({r['text'] for r in rows}),
            'documents': len({r['doc_id'] for r in rows}),
            'mean_characters': sum(r['characters'] for r in rows) / n}


def paired_summary(pairs, *, bootstrap=0, seed=1234):
    if not pairs:
        return {'n': 0}
    words = [p['word'] for p in pairs]
    arbitrary = [p['arbitrary'] for p in pairs]
    n = len(pairs)
    delta = sum(int(w['exact']) - int(a['exact']) for w, a in zip(words, arbitrary)) / n
    result = {'n': n, 'word': aggregate(words), 'arbitrary': aggregate(arbitrary),
              'word_minus_arbitrary_em': delta,
              'word_only_exact': sum(w['exact'] and not a['exact'] for w, a in zip(words, arbitrary)),
              'arbitrary_only_exact': sum(a['exact'] and not w['exact'] for w, a in zip(words, arbitrary)),
              'identical_span_pairs': sum(w['start_byte'] == a['start_byte'] for w, a in zip(words, arbitrary)),
              'identical_text_pairs': sum(w['text'] == a['text'] for w, a in zip(words, arbitrary)),
              'arbitrary_on_legacy_span': sum(a['is_legacy_word_span'] for a in arbitrary)}
    if bootstrap:
        by_doc = defaultdict(lambda: [0, 0])
        for w, a in zip(words, arbitrary):
            by_doc[w['doc_id']][0] += int(w['exact']) - int(a['exact'])
            by_doc[w['doc_id']][1] += 1
        values = np.array(list(by_doc.values()), dtype=float)
        rng = np.random.default_rng(seed)
        boot = []
        for _ in range(bootstrap):
            drawn = values[rng.integers(len(values), size=len(values))].sum(0)
            boot.append(drawn[0] / drawn[1])
        result['document_bootstrap_delta95'] = np.quantile(boot, [.025, .975]).tolist()
        result['bootstrap_replicates'] = bootstrap
    return result


def summarize(natural, pairs):
    def length(row):
        return str(row['bytes'])
    def bucket(row):
        low = ((row['bytes'] - 1) // 8) * 8 + 1
        return f'{low:02d}-{low+7:02d}'
    def group(rows, key, fn):
        result = defaultdict(list)
        for row in rows:
            result[key(row)].append(row)
        return {k: fn(v) for k, v in sorted(result.items())}
    def pair_views(subset):
        return {'overall': paired_summary(subset, bootstrap=300),
                'by_byte_length': group(subset, length, paired_summary),
                'by_8byte_bucket': group(subset, bucket, lambda ps: paired_summary(ps, bootstrap=300)),
                'by_language': group(subset, lambda p: p['language'], lambda ps: paired_summary(ps, bootstrap=300)),
                'by_language_and_length': group(subset, lambda p: p['language'],
                                                 lambda ps: group(ps, length, paired_summary))}
    return {'natural_stratified': {'overall': aggregate(natural),
                                  'by_byte_length': group(natural, length, aggregate),
                                  'by_8byte_bucket': group(natural, bucket, aggregate),
                                  'by_language_and_length': group(natural, lambda p: p['language'],
                                                                  lambda rows: group(rows, length, aggregate))},
            'matched_all_legacy_units': pair_views(pairs),
            'matched_lexical_legacy_units': pair_views([p for p in pairs if p['word']['contains_letter']])}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--per-cell', type=int, default=256)
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--seed', type=int, default=1234)
    args = parser.parse_args()
    if args.per_cell < 1:
        raise ValueError('per-cell must be positive')
    torch.set_num_threads(4)
    started = time.perf_counter()
    records = [json.loads(line) for line in args.data.read_text(encoding='utf-8').splitlines() if line]
    natural, pairs, segmentation, support = build_samples(records, per_cell=args.per_cell, seed=args.seed)
    print(json.dumps({'stage': 'sampled', 'natural': len(natural), 'pairs': len(pairs),
                      'seconds': time.perf_counter() - started}), flush=True)
    rows = natural + [r for p in pairs for r in (p['word'], p['arbitrary'])]
    unique = list(dict.fromkeys(r['text'].encode('utf-8') for r in rows))
    with args.checkpoint.open('rb') as handle:
        checkpoint_sha = hashlib.sha256(handle.read()).hexdigest()
        handle.seek(0)
        saved = torch.load(handle, map_location='cpu', weights_only=False)
    ae = ByteStringAE(Cfg(saved['cfg']))
    ae.load_state_dict(saved['model'], strict=True)
    ae.requires_grad_(False)
    scorer = AEQualityScorer(ae, device=args.device, batch_size=args.batch_size)
    scores = scorer.score_records(unique)
    lookup = dict(zip(unique, scores))
    for row in rows:
        score = lookup[row['text'].encode('utf-8')]
        row.update(exact=score['feasible'], eos_exact=score['eos_argmax'], target_nll_nats=score['target_nll_nats'])
    result = {'checkpoint_sha256': checkpoint_sha, 'checkpoint_step': saved.get('step'),
              'training_state': saved.get('state'), 'data_file_sha256': sha(args.data),
              'script_sha256': sha(__file__),
              'segmenter_sha256': sha(Path(__file__).resolve().parents[1] / 'char_diff/vendor/bltz/segment_v2.py'),
              'protocol': {'seed': args.seed, 'per_cell': args.per_cell, 'precision': 'fp32',
                           'device': args.device, 'batch_size': args.batch_size, 'split': 'calibration',
                           'em': 'every byte and true EOS argmax; no verification/resplitting/filter by reconstruction',
                           'natural': 'each language x exact byte length: uniform feasible document then uniform valid start, with replacement',
                           'word': 'legacy segment_word merge=prev, raw units <=32B, occurrence reservoir without replacement per language x byte length',
                           'matched_arbitrary': 'same document and exact byte length, uniform UTF-8-valid start, no augmentation',
                           'over32': 'raw legacy runs excluded from primary comparison; counts and byte coverage reported',
                           'uncertainty': 'Wilson intervals describe per-row proportions; paired differences include 300 document-cluster bootstrap replicates',
                           'multilingual_legacy': 'Unicode letter runs; CJK runs follow the old implementation, not dictionary-based Chinese word segmentation'},
              'segmentation_coverage': segmentation, 'support': support,
              'unique_model_inputs': len(unique), 'elapsed_seconds': time.perf_counter() - started,
              **summarize(natural, pairs)}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, values in [('natural.jsonl', natural), ('matched_pairs.jsonl', pairs)]:
        with (args.output_dir / name).open('w', encoding='utf-8') as handle:
            for row in values:
                handle.write(json.dumps(row, ensure_ascii=False, separators=(',', ':')) + '\n')
    write_json(args.output_dir / 'results.json', result)
    print(json.dumps({'stage': 'complete', 'unique_model_inputs': len(unique),
                      'seconds': time.perf_counter() - started,
                      'paired': result['matched_all_legacy_units']['overall']}), flush=True)


if __name__ == '__main__':
    main()
