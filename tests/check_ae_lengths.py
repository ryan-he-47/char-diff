"""Only the new length matching and aggregation paths."""
import importlib.util
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location('ae_lengths', ROOT / 'scripts/evaluate_ae_lengths.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


class LengthChecks(unittest.TestCase):
    def test_utf8_same_document_same_length_and_determinism(self):
        records = [
            {'id': 'one', 'language': 'english', 'split': 'calibration', 'text': 'Hello! ' + 'a' * 36 + ' tiny stories.'},
            {'id': 'two', 'language': 'chinese', 'split': 'calibration', 'text': '你好，世界。' + '猫' * 12 + ' A🙂!'},
        ]
        first = m.build_samples(records, per_cell=4)
        self.assertEqual(first, m.build_samples(records, per_cell=4))
        natural, pairs, coverage, support = first
        self.assertEqual(coverage['english']['over32_units'], 1)
        self.assertEqual(coverage['chinese']['over32_units'], 1)
        by_id = {r['id']: r['text'].encode() for r in records}
        for row in natural + [s for p in pairs for s in (p['word'], p['arbitrary'])]:
            unit = by_id[row['doc_id']][row['start_byte']:row['end_byte']]
            self.assertEqual(unit.decode(), row['text'])
            self.assertEqual(len(unit), row['bytes'])
        for pair in pairs:
            w, a = pair['word'], pair['arbitrary']
            self.assertEqual((w['doc_id'], w['language'], w['bytes']), (a['doc_id'], a['language'], a['bytes']))
            self.assertTrue(w['is_legacy_word_span'])
        self.assertTrue(any(s['matched_pairs'] == 0 for s in support))

    def test_paired_aggregation_and_cluster_interval(self):
        base = {'doc_id': 'a', 'start_byte': 0, 'end_byte': 2, 'bytes': 2, 'text': 'ab',
                'characters': 2, 'target_nll_nats': 0., 'is_legacy_word_span': True}
        pairs = [
            {'word': dict(base, exact=True), 'arbitrary': dict(base, exact=False, start_byte=3)},
            {'word': dict(base, doc_id='b', exact=True), 'arbitrary': dict(base, doc_id='b', exact=True)},
        ]
        out = m.paired_summary(pairs, bootstrap=300)
        self.assertEqual(out['word']['em'], 1.)
        self.assertEqual(out['arbitrary']['em'], .5)
        self.assertEqual(out['word_minus_arbitrary_em'], .5)
        self.assertEqual(out['word_only_exact'], 1)
        self.assertEqual(out['identical_span_pairs'], 1)
        self.assertEqual(out['document_bootstrap_delta95'], [0., 1.])


if __name__ == '__main__':
    unittest.main(verbosity=2)
