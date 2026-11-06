"""Targeted checks for frozen path-latent diagnostics only."""
import importlib.util
from pathlib import Path
import random
import sys
import unittest

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location('path_latents', ROOT / 'scripts/analyze_path_latents.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


class ToyAE:
    l_max = 3

    def encoder(self, ids, mask):
        return ids[:, :2].float()

    def decoder(self, z, delta):
        logits = torch.full((len(z), 257), -20., device=z.device)
        selected = torch.where(delta < 2, z[torch.arange(len(z)), delta.clamp_max(1)].round().long(), 256)
        logits[torch.arange(len(z)), selected] = 20
        return logits


class Checks(unittest.TestCase):
    def test_edits_match_dynamic_programming(self):
        rng = random.Random(42)
        for _ in range(100):
            a, b = [bytes(rng.randrange(5) for _ in range(rng.randrange(33))) for _ in range(2)]
            dp = list(range(len(b) + 1))
            for i, x in enumerate(a):
                nxt = [i + 1]
                for j, y in enumerate(b):
                    nxt.append(min(dp[j + 1] + 1, nxt[-1] + 1, dp[j] + int(x != y)))
                dp = nxt
            self.assertEqual(m.edit_distance(a, b), dp[-1])
        self.assertEqual(m.edit_distance('小猫🙂', '小狗🙂'), 1)

    def test_geometry_and_stratified_weights(self):
        z = np.array([[1., 0], [-1, 0], [0, 1], [0, -1]])
        stats = m.geometry(z, [b'a', b'bb', b'c', b'dd'], [1, 2, 1, 2])
        self.assertAlmostEqual(stats['participation_ratio'], 2.)
        self.assertAlmostEqual(stats['entropy_effective_rank'], 2.)
        rows = [{'language': 'en', 'bytes': 1}] * 25 + [{'language': 'zh', 'bytes': 3}] * 5
        ids, weights = m.stratified_indices(rows, 10)
        self.assertEqual(len(ids), 10)
        self.assertEqual(sum(weights), 30)
        self.assertEqual(sum(w for i, w in zip(ids, weights) if i >= 25), 5)
        projections = m.pca_projection(z, z, [1, 2])
        np.testing.assert_array_equal(projections[2], z.astype(np.float32))
        self.assertEqual(np.linalg.matrix_rank(projections[1]), 1)

    def test_noise_clean_eos_and_cycle(self):
        rows = [{'hex': b'ab'.hex(), 'bytes': 2, 'characters': 2, 'language': 'en'}]
        z = np.array([[97., 98.]], dtype=np.float32)
        clean = m.perturb(ToyAE(), z, rows, [3], 2., 0., 42, 'cpu', 2)
        self.assertTrue(clean[0]['exact'])
        self.assertEqual(clean[0]['cycle_rmse'], 0.)
        result = m.summarize_noise(clean)
        self.assertEqual(result['em_unit_weighted'], 1.)
        self.assertEqual(result['em_byte_weighted'], 1.)
        self.assertEqual(result['byte_edit_rate'], 0.)
        # The same string receives the same perturbation under another path order.
        a = m.perturb(ToyAE(), z, rows, [1], 2., .025, 42, 'cpu', 2)
        b = m.perturb(ToyAE(), np.repeat(z, 2, 0), rows * 2, [1, 1], 2., .025, 42, 'cpu', 2)
        self.assertEqual(a[0]['cycle_rmse'], b[1]['cycle_rmse'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
