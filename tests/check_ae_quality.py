"""Targeted EOS, aggregation, UTF-8 and real-model checks for AE scores."""
from pathlib import Path
import math
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch

from char_diff.ae_paths import shortest_path
from char_diff.ae_quality import AEQualityScorer, tensorize, targets_from_lengths
from char_diff.vendor.bltz.config import Cfg
from char_diff.vendor.bltz.models.autoencoder import ByteStringAE


class ExactLogits(torch.nn.Module):
    def __init__(self, wrong_max_eos=False):
        super().__init__()
        self.l_max = 32
        self.wrong_max_eos = wrong_max_eos

    def forward(self, ids, pad_mask):
        lens = (~pad_mask).sum(-1)
        targets = targets_from_lengths(ids, lens).clamp_min(0)
        logits = torch.zeros((len(ids), 33, 257), device=ids.device)
        logits.scatter_(-1, targets[..., None], 3.)
        if self.wrong_max_eos:
            for row, length in enumerate(lens.tolist()):
                if length == 32:
                    logits[row, 32, 0] = 4.
        return logits


class QualityChecks(unittest.TestCase):
    def test_full32_checks_eos(self):
        ae = ExactLogits(wrong_max_eos=True)
        ae.train()
        scorer = AEQualityScorer(ae)
        records = scorer.score_records([b"a" * 32, b"a" * 31])
        self.assertFalse(records[0]["feasible"])
        self.assertFalse(records[0]["eos_argmax"])
        self.assertTrue(records[1]["feasible"])
        self.assertTrue(ae.training)

    def test_aggregation_and_utf8(self):
        units = [b"\0a", "汉🙂".encode()]
        ids, lens, mask = tensorize(units, 32, "cpu")
        targets = targets_from_lengths(ids, lens)
        self.assertEqual((targets != -100).sum(-1).tolist(), [3, 8])
        self.assertEqual(targets[0, :4].tolist(), [0, 97, 256, -100])
        scorer = AEQualityScorer(ExactLogits())
        records = scorer.score_records(units)
        p_true = math.exp(3) / (math.exp(3) + 256)
        p_other = 1 / (math.exp(3) + 256)
        one_nll = -math.log(p_true)
        one_entropy = -p_true * math.log(p_true) - 256 * p_other * math.log(p_other)
        for unit, record in zip(units, records):
            self.assertTrue(record["feasible"])
            self.assertAlmostEqual(record["target_nll_nats"], (len(unit) + 1) * one_nll, places=4)
            self.assertAlmostEqual(record["entropy_nats"], (len(unit) + 1) * one_entropy, places=4)
            self.assertAlmostEqual(record["mean_entropy_nats"], one_entropy, places=5)
            self.assertAlmostEqual(record["min_logit_margin"], 3., places=6)
        q = AEQualityScorer(ExactLogits(), quality_kind="entropy")(units)
        self.assertAlmostEqual(q[0].quality, records[0]["entropy_nats"], places=5)
        with self.assertRaises(UnicodeDecodeError): tensorize([b"\xe6\xb1"], 32, "cpu")
        with self.assertRaises(ValueError): tensorize([b""], 32, "cpu")
        with self.assertRaises(ValueError): tensorize([b"a" * 33], 32, "cpu")

    def test_existing_ae_small_path(self):
        checkpoint = Path(__file__).resolve().parents[1] / "assets/ae_best.pt"
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        ae = ByteStringAE(Cfg(state["cfg"]))
        ae.load_state_dict(state["model"], strict=True)
        scorer = AEQualityScorer(ae, batch_size=8)
        raw = b"cat"
        result = shortest_path(raw, scorer=scorer, batch_size=8, quality_weight=.1)
        self.assertEqual(b"".join(result.units), raw)
        for row in scorer.score_records(result.units):
            self.assertTrue(row["feasible"])
            self.assertTrue(row["eos_argmax"])


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main(verbosity=2)
