"""Only new risk aggregation and latent-gradient paths."""
import importlib.util
from pathlib import Path
import sys
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
spec=importlib.util.spec_from_file_location('risk',ROOT/'scripts/analyze_candidate_risks.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
import torch


class Checks(unittest.TestCase):
    def test_auc_ties_replicates_and_length_strata(self):
        self.assertEqual(m.weighted_auc([0,1],[0,2],[2,0])['auc'],1.)
        self.assertEqual(m.weighted_auc([0,0],[0,2],[2,0])['auc'],.5)
        # Both length strata contain failures and successes; length itself is constant inside each.
        rows=[{'language':'en','bytes':length,'risk_scores':{'length':float(length)},
               'failures':{'0.1':{'count':pos,'trials':2}}}
              for length,pos in [(1,0),(1,1),(2,1),(2,2)]]
        result=m.summarize_auc(rows,'length','0.1')
        self.assertGreater(result['global']['auc'],.5)
        self.assertEqual(result['same_language_and_exact_byte_length']['auc'],.5)
        self.assertEqual(result['same_language_and_exact_byte_length']['positive_negative_pairs'],6)

    def test_local_gradient_radius_in_standardized_coordinates(self):
        class Decoder(torch.nn.Module):
            def __init__(self):
                super().__init__();self.weight=torch.nn.Parameter(torch.tensor(1.),requires_grad=False)
            def forward(self,z,position):
                logits=torch.full((len(z),257),-100.,device=z.device)
                logits[:,1]=0.
                label=torch.where(position==0,0,256)
                value=z[:,0]*self.weight*(position+1)
                return logits.scatter(1,label[:,None],value[:,None])
        class AE(torch.nn.Module):
            def __init__(self):
                super().__init__();self.l_max=1;self.decoder=Decoder()
        ae=AE();z=torch.tensor([[2.,8.],[3.,7.]])
        result=m.weakest_margin_gradient(ae,z,torch.tensor([[0,256],[0,256]]),2.)
        self.assertEqual([row['local_radius_standardized_l2'] for row in result],[1.,1.5])
        self.assertEqual([row['weak_position'] for row in result],[0,0])
        self.assertTrue(all(row['clean_exact'] for row in result))
        self.assertIsNone(z.grad);self.assertFalse(z.requires_grad)
        self.assertIsNone(ae.decoder.weight.grad);self.assertFalse(ae.decoder.weight.requires_grad)


if __name__=='__main__':unittest.main(verbosity=2)
