"""Score complete UTF-8 candidate graphs once and compare exploratory paths."""
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
import torch
from tokenizers import Tokenizer

from char_diff.ae_paths import boundary_offsets, candidate_spans, NoFeasiblePath
from char_diff.ae_path_variants import CandidateGraph, default_variants
from char_diff.ae_quality import AEQualityScorer
from char_diff.utils import write_json
from char_diff.vendor.bltz.config import Cfg
from char_diff.vendor.bltz.models.autoencoder import ByteStringAE
from char_diff.vendor.bltz.segment_v2 import segment_word


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prior_boundaries(raw, tokenizer):
    """Apply priors to original text; only UTF-8 legal endpoints enter the DAG."""
    text = raw.decode('utf-8')
    offsets = [0]
    for ch in text:
        offsets.append(offsets[-1] + len(ch.encode('utf-8')))
    allowed = set(offsets) - {0, len(raw)}

    def mapped(spans):
        result = set()
        for start, end in spans:
            if not (0 <= start <= end <= len(text)):
                raise ValueError('Tokenizer offsets outside original codepoint range')
            result.add(offsets[start])
            result.add(offsets[end])
        return result & allowed

    encoded = tokenizer.encode(text, add_special_tokens=False)
    bpe = mapped(encoded.offsets)
    regex = mapped([offset for _, offset in tokenizer.pre_tokenizer.pre_tokenize_str(text)])
    units = segment_word(raw, '', l_max=max(32, len(raw)), merge='prev')
    assert b''.join(units) == raw
    pos, word = 0, set()
    for unit in units:
        pos += len(unit)
        if pos != len(raw):
            word.add(pos)
    return {'regex': sorted(regex), 'bpe': sorted(bpe), 'word': sorted(word)}, {
        'bpe_token_count': len(encoded.ids),
        'normalizer_changes_text': bool(tokenizer.normalizer and tokenizer.normalizer.normalize_str(text) != text),
        'regex_input': 'original text, tokenizer.pre_tokenizer only',
        'bpe_input': 'original text through full tokenizer; original offsets mapped to UTF-8 boundaries',
    }


def boundary_metrics(boundaries, priors):
    cuts = set(boundaries[1:-1])
    return {name: {'selected_cuts': len(cuts), 'prior_cuts': len(preferred),
                   'hits': len(cuts & set(preferred)),
                   'extra_cuts': len(cuts - set(preferred)),
                   'missed_cuts': len(set(preferred) - cuts),
                   'symmetric_difference': len(cuts ^ set(preferred))}
            for name, preferred in priors.items()}


def run(checkpoint, documents, output_dir, *, checkpoint_tag, tokenizer_path,
        config=None, batch_size=512, device='cuda', compress=False):
    started = time.perf_counter()
    torch.set_num_threads(4)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = '.jsonl.gz' if compress else '.jsonl'
    paths_file = output_dir / ('paths' + suffix)
    samples_file = output_dir / ('candidate_samples' + suffix)
    if paths_file.exists() or samples_file.exists():
        raise FileExistsError('Use a new output directory to preserve earlier path evidence')
    variants = json.loads(Path(config).read_text(encoding='utf-8')) if config else default_variants()
    if isinstance(variants, dict):
        variants = variants['variants']
    names = [v['name'] for v in variants]
    if len(names) != len(set(names)):
        raise ValueError('Variant names must be unique')
    for i, spec in enumerate(variants):
        if spec.get('same_k_as') and spec['same_k_as'] not in names[:i]:
            raise ValueError('same_k_as must refer to an earlier variant')
    with Path(checkpoint).open('rb') as handle:
        checkpoint_sha = hashlib.sha256(handle.read()).hexdigest()
        handle.seek(0)
        saved = torch.load(handle, map_location='cpu', weights_only=False)
    ae = ByteStringAE(Cfg(saved['cfg']))
    ae.load_state_dict(saved['model'], strict=True)
    ae.requires_grad_(False)
    scorer = AEQualityScorer(ae, device=device, batch_size=batch_size)
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    inputs = [json.loads(line) for line in Path(documents).read_text(encoding='utf-8').splitlines() if line]
    counters = defaultdict(lambda: {'documents': 0, 'covered_documents': 0, 'source_bytes': 0,
                                   'covered_bytes': 0, 'units': 0, 'single_character_units': 0,
                                   'total_nll': 0., 'total_byte_nll': 0., 'total_eos_nll': 0.,
                                   'total_entropy': 0., 'byte_risk_sum': 0., 'margin_risk_sum': 0.,
                                   'lengths': [], 'boundary': defaultdict(lambda: defaultdict(int))})
    opener = gzip.open if compress else open
    with opener(paths_file, 'wt', encoding='utf-8', newline='\n') as paths, \
            opener(samples_file, 'wt', encoding='utf-8', newline='\n') as samples:
        for doc_number, doc in enumerate(inputs):
            raw = bytes.fromhex(doc['source_hex']) if 'source_hex' in doc else doc['text'].encode('utf-8')
            raw.decode('utf-8', errors='strict')
            if 'text' in doc and doc['text'].encode('utf-8') != raw:
                raise ValueError('Panel text/source_hex mismatch')
            doc_id = str(doc.get('doc_id', doc.get('id')))
            doc_start = time.perf_counter()
            priors, prior_info = prior_boundaries(raw, tokenizer)
            candidates = list(dict.fromkeys(raw[a:b] for a, b in candidate_spans(raw, max_bytes=ae.l_max)))
            score_rows = scorer.score_records(candidates)
            for row in score_rows:
                probability = row['eos_probability']
                row['eos_nll_nats'] = -math.log(probability) if probability > 0 else None
                row['target_byte_nll_nats'] = (row['target_nll_nats'] - row['eos_nll_nats']) if probability > 0 else None
            scores = dict(zip(candidates, score_rows))
            del score_rows, candidates
            graph = CandidateGraph(raw, scores, max_bytes=ae.l_max)
            common = {'checkpoint_tag': checkpoint_tag, 'doc_id': doc_id, 'language': doc['language'],
                      'fold': doc.get('fold', 'analysis'), 'source_hex': raw.hex(),
                      'source_bytes': len(raw), 'unique_candidates': len(scores),
                      'candidate_occurrences': graph.candidate_occurrences,
                      'feasible_candidate_occurrences': graph.feasible_occurrences,
                      'prior_boundaries': priors, 'prior_info': prior_info}
            # Same unfiltered unique candidate sample across checkpoints.
            selected = sorted(scores, key=lambda u: hashlib.sha256(doc_id.encode() + b'\0' + u).digest())[:64]
            for unit in selected:
                samples.write(json.dumps({'checkpoint_tag': checkpoint_tag, 'doc_id': doc_id,
                                          'language': doc['language'], 'fold': doc.get('fold', 'analysis'),
                                          **scores[unit]}, ensure_ascii=False, allow_nan=False) + '\n')
            solved = {}
            for spec in variants:
                row = {**common, 'variant': spec['name'], 'spec': spec}
                reference = solved.get(spec.get('same_k_as', ''))
                try:
                    if spec.get('same_k_as') and not reference:
                        raise NoFeasiblePath('Reference path unavailable')
                    exact_k = reference['k'] if reference else spec.get('exact_k')
                    if spec.get('random'):
                        seed = int.from_bytes(hashlib.sha256((doc_id + spec['name']).encode()).digest()[:8], 'big')
                        result = graph.random_fixed_k(exact_k, seed)
                    else:
                        result = graph.solve(spec, prior_boundaries=priors.get(spec.get('prior'), ()), exact_k=exact_k)
                    result['exact_k_target'] = exact_k
                    result['boundary_metrics'] = boundary_metrics(result['boundaries'], priors)
                    row.update(result)
                    solved[spec['name']] = result
                except NoFeasiblePath as exc:
                    row.update(coverage=False, error=str(exc), k=None, units=[])
                paths.write(json.dumps(row, ensure_ascii=False, allow_nan=False, separators=(',', ':')) + '\n')
                for group in ['all', f'fold:{row["fold"]}', f'language:{row["language"]}']:
                    count = counters[(spec['name'], group)]
                    count['documents'] += 1
                    count['source_bytes'] += len(raw)
                    if row['coverage']:
                        count['covered_documents'] += 1
                        count['covered_bytes'] += len(raw)
                        count['units'] += row['k']
                        count['single_character_units'] += sum(len(bytes.fromhex(u['hex']).decode('utf-8')) == 1 for u in row['units'])
                        count['lengths'].extend(u['bytes'] for u in row['units'])
                        for key in ['total_nll', 'total_byte_nll', 'total_eos_nll', 'total_entropy', 'byte_risk_sum', 'margin_risk_sum']:
                            count[key] += row[key]
                        for name, vals in row['boundary_metrics'].items():
                            for key, value in vals.items():
                                count['boundary'][name][key] += value
            paths.flush()
            print(json.dumps({'doc': doc_number + 1, 'of': len(inputs), 'language': doc['language'],
                              'source_bytes': len(raw), 'unique_candidates': len(scores),
                              'covered_variants': len(solved), 'seconds': time.perf_counter() - doc_start}), flush=True)
    for count in counters.values():
        lengths = count.pop('lengths')
        count['length_histogram'] = {str(n): lengths.count(n) for n in range(1, 33)}
        count['bytes_per_unit'] = count['covered_bytes'] / count['units'] if count['units'] else None
        count['nll_per_source_byte'] = count['total_nll'] / count['covered_bytes'] if count['covered_bytes'] else None
        count['single_character_fraction'] = count['single_character_units'] / count['units'] if count['units'] else None
    grouped = defaultdict(dict)
    for (name, group), count in counters.items():
        grouped[name][group] = count
    manifest = {'checkpoint_tag': checkpoint_tag, 'checkpoint_sha256': checkpoint_sha,
                'checkpoint_step': saved.get('step'), 'documents_sha256': sha(documents),
                'tokenizer_sha256': sha(tokenizer_path), 'script_sha256': sha(__file__),
                'module_sha256': sha(Path(__file__).resolve().parents[1] / 'char_diff/ae_path_variants.py'),
                'precision': 'fp32', 'device': device, 'batch_size': batch_size,
                'variants': variants, 'documents': len(inputs), 'elapsed_seconds': time.perf_counter() - started,
                'paths': str(paths_file), 'paths_sha256': sha(paths_file),
                'candidate_samples': str(samples_file), 'candidate_samples_sha256': sha(samples_file),
                'candidate_sampling': '64 smallest SHA256(doc_id NUL unique_unit) per document, before feasibility filtering',
                'prior_cost': 'symmetric difference of selected internal UTF8 cuts and prior internal cuts',
                'quality_definitions': {'nll': 'total target byte+EOS NLL', 'entropy': 'total position entropy including EOS',
                                        'byte_risk': 'bytes * (1-exp(-total target NLL)); factorized softmax risk proxy',
                                        'margin_risk': 'bytes * softplus(-minimum target-vs-runnerup logit margin)'},
                'normalization': 'No source text normalization; BPE tokenizer has its own stored normalizer; original offsets retained'}
    write_json(output_dir / 'manifest.json', manifest)
    write_json(output_dir / 'summary.json', dict(grouped))
    print(json.dumps({'complete': True, 'seconds': manifest['elapsed_seconds'], 'paths': str(paths_file)}), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--checkpoint-tag', required=True)
    p.add_argument('--documents', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--tokenizer', dest='tokenizer_path', type=Path, default=Path('assets/tokenizer.json'))
    p.add_argument('--config', type=Path)
    p.add_argument('--batch-size', type=int, default=512)
    p.add_argument('--device', default='cuda')
    p.add_argument('--compressed', dest='compress', action='store_true')
    run(**vars(p.parse_args()))


if __name__ == '__main__':
    main()
