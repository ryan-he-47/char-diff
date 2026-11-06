"""CPU source check with no dataset, codec checkpoint, or training job."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from char_diff.model import Denoiser

torch.set_num_threads(2)
torch.manual_seed(42)
cfg = dict(width=64, latent_dim=48, heads=4, ffn=128, layers=2,
           block=4, grad_checkpoint=False)
model = Denoiser(cfg).eval()
# Make attention and its output observable: the production constructor starts
# with zero output weights, which would hide a leakage bug in a smoke check.
with torch.no_grad():
    model.output.weight.normal_(std=.02)
    for block in model.blocks:
        block.mod.bias[2*cfg['width']:3*cfg['width']].fill_(1.)
        block.mod.bias[5*cfg['width']:6*cfg['width']].fill_(1.)
x = torch.randn(1, 8, 48)
t = torch.full((1, 8), .5)
roles = torch.ones(1, 8, dtype=torch.long)
positions = torch.arange(8)[None]
lengths = torch.full_like(positions, 4)
group = positions // 4
allowed = (group[:, :, None] == group[:, None, :])[:, None]
with torch.inference_mode():
    output = model(x, t, roles, positions, lengths, allowed)
    changed = x.clone(); changed[:, 4:] += 100
    again = model(changed, t, roles, positions, lengths, allowed)
assert output.shape == (1, 8, 48) and torch.isfinite(output).all()
torch.testing.assert_close(output[:, :4], again[:, :4], atol=1e-6, rtol=0)
assert (output[:, 4:] - again[:, 4:]).abs().max() > 1e-5
print('PASS: imports, finite CPU forward, future-block isolation; no data or weights required')
