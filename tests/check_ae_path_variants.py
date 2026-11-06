"""Targeted checks for exploratory DAG costs, fixed-K and boundary priors."""
import importlib.util
import itertools
import math
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from char_diff.ae_paths import candidate_spans, NoFeasiblePath
from char_diff.ae_path_variants import CandidateGraph
from tokenizers import Tokenizer
spec = importlib.util.spec_from_file_location('path_eval', ROOT / 'scripts/evaluate_ae_paths.py')
evaluation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluation)


def scores(raw, max_bytes):
    return {raw[a:b]: {'bytes': b-a, 'unit_hex': raw[a:b].hex(), 'feasible': True,
                       'target_nll_nats': .05*(b-a)**2, 'entropy_nats': .2*(b-a),
                       'min_logit_margin': 1.5, 'eos_probability': .99}
            for a,b in candidate_spans(raw, max_bytes=max_bytes)}


class Checks(unittest.TestCase):
    def test_unconstrained_and_exact_k_match_exhaustive(self):
        raw=b'abcde'
        mapping=scores(raw,3);mapping[b'bc']['feasible']=False
        graph=CandidateGraph(raw,mapping,max_bytes=3)
        spec={'quality':'nll','weight':2.,'prior':'regex','prior_weight':.3}
        prior={2,4}
        brute=[]
        for mask in itertools.product([0,1], repeat=4):
            boundaries=[0]+[i+1 for i,v in enumerate(mask) if v]+[5]
            units=[raw[a:b] for a,b in zip(boundaries,boundaries[1:])]
            if any(u not in mapping or not mapping[u]['feasible'] for u in units):continue
            cost=len(units)+2*sum(mapping[u]['target_nll_nats'] for u in units)+.3*len(set(boundaries[1:-1])^prior)
            brute.append((len(units),cost))
        result=graph.solve(spec,prior_boundaries=prior)
        self.assertAlmostEqual(result['cost'],min(cost for _,cost in brute))
        for k in {k for k,_ in brute}:
            result=graph.solve(spec,prior_boundaries=prior,exact_k=k)
            self.assertEqual(result['k'],k)
            self.assertAlmostEqual(result['cost'],min(cost for count,cost in brute if count==k))
        with self.assertRaises(NoFeasiblePath):graph.solve(spec,exact_k=1)

    def test_fixed_k_prior_and_uniform_random_control(self):
        raw=b'abc';graph=CandidateGraph(raw,scores(raw,2),max_bytes=2)
        spec={'quality':'none','weight':0.,'prior':'regex','prior_weight':1.}
        a=graph.solve(spec,prior_boundaries={1},exact_k=2)
        b=graph.solve(spec,prior_boundaries={2},exact_k=2)
        self.assertEqual(a['boundaries'],[0,1,3]);self.assertEqual(b['boundaries'],[0,2,3])
        self.assertEqual(a['cost'],2.);self.assertEqual(b['cost'],2.)
        draw=graph.random_fixed_k(2,42)
        self.assertEqual(draw,graph.random_fixed_k(2,42))
        self.assertEqual(draw['k'],2)
        self.assertAlmostEqual(draw['uniform_exact_k_paths_log_count'],math.log(2))

    def test_utf8_prior_offsets_with_nfc_and_combining_marks(self):
        tokenizer=Tokenizer.from_file(str(ROOT/'assets/tokenizer.json'))
        raw='Cafe\u0301 猫咪123! 👩‍💻'.encode('utf8')
        priors,info=evaluation.prior_boundaries(raw,tokenizer)
        self.assertTrue(info['normalizer_changes_text'])
        for boundaries in priors.values():
            for b in boundaries:
                raw[:b].decode('utf8');raw[b:].decode('utf8')
        self.assertGreater(len(priors['regex']),0)
        self.assertGreater(len(priors['bpe']),0)


if __name__=='__main__':unittest.main(verbosity=2)
