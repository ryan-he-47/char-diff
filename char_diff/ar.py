"""Causal latent LM with the legacy linear, learned-diagonal GMM likelihood."""
from __future__ import annotations
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from .data import LatentCache
from .model import Attention, RMSNorm
from .vendor.bltz.models.mdn import MDNHead


class ARCache(LatentCache):
    """Reuse exactly the packed windows/targets; shift within each story segment."""
    def batch(self, indices, device="cpu"):
        layouts = []
        for index in indices:
            items = []
            for doc_number, part in enumerate(self.segments[int(self.windows[index]):int(self.windows[index+1])]):
                doc, st, en, pr = map(int, part)
                base = int(self.offsets[doc])
                for q in range(pr, en):
                    previous = np.zeros(48, np.float32) if q == pr else self.vectors[int(self.ids[base+q-1])]
                    target = self.vectors[int(self.ids[base+q])]
                    items.append((previous, target, q, doc_number, q == pr, q >= st))
            layouts.append(items)
        b, n = len(layouts), max(map(len, layouts))
        x = np.zeros((b, n, 48), np.float32); target = np.zeros_like(x)
        positions = np.zeros((b, n), np.int64); docs = np.full((b, n), -1, np.int64)
        bos = np.zeros((b, n), bool); mask = np.zeros((b, n), bool)
        for i, items in enumerate(layouts):
            for j, (previous, y, pos, doc, begin, valid) in enumerate(items):
                x[i,j] = previous; target[i,j] = y; positions[i,j] = pos
                docs[i,j] = doc; bos[i,j] = begin; mask[i,j] = valid
        batch = {k: torch.from_numpy(v).to(device) for k, v in
                 dict(x=x, target=target, positions=positions, docs=docs, bos=bos, mask=mask).items()}
        d = batch['docs']; p = batch['positions']
        allowed = (d[:,:,None] == d[:,None,:]) & (d[:,:,None] >= 0) & (p[:,None,:] <= p[:,:,None])
        allowed |= (d < 0)[:,:,None] & torch.eye(n, device=device, dtype=torch.bool)[None]
        batch['allowed'] = allowed[:,None]
        return batch


class ARBlock(nn.Module):
    def __init__(self, width, heads, ffn):
        super().__init__()
        self.norm1, self.norm2 = RMSNorm(width), RMSNorm(width)
        self.attn = Attention(width, heads)
        self.up = nn.Linear(width, 2*ffn, bias=False)
        self.down = nn.Linear(ffn, width, bias=False)

    def forward(self, h, positions, allowed, past=None):
        a, kv = self.attn(self.norm1(h), positions, allowed, past)
        h = h.float() + a.float()
        u, v = self.up(self.norm2(h)).chunk(2, -1)
        h = h + self.down(F.silu(u)*v).float()
        return h, kv


class FP32MDNHead(MDNHead):
    def params(self, h):
        # Density arithmetic and the small readout stay FP32 under outer AMP.
        with torch.autocast(device_type=h.device.type, enabled=False):
            return super().params(h.float())


class GMMAR(nn.Module):
    def __init__(self, cfg):
        super().__init__(); self.cfg = dict(cfg)
        w, d = cfg['width'], cfg.get('latent_dim', 48)
        if w % cfg['heads'] or (w//cfg['heads']) % 2:
            raise ValueError('RoPE requires even head width')
        if cfg['components'] < 1 or cfg['sigma_floor'] <= 0:
            raise ValueError('Require positive component count and sigma floor')
        self.input = nn.Linear(d, w, bias=False)
        self.bos = nn.Parameter(torch.randn(w)*.02)
        self.blocks = nn.ModuleList([ARBlock(w, cfg['heads'], cfg['ffn']) for _ in range(cfg['layers'])])
        self.norm = RMSNorm(w)
        self.head = FP32MDNHead(d_in=w, n_comp=cfg['components'], d_emb=d,
                               sigma_floor=cfg['sigma_floor'], depth=0)

    def forward(self, x, positions, bos, allowed, past=None, return_cache=False):
        h = self.input(x).float()
        h = torch.where(bos[...,None], self.bos, h)
        caches = []
        for i, block in enumerate(self.blocks):
            if self.training and self.cfg.get('grad_checkpoint', False) and not return_cache:
                def run(value, layer=block):
                    return layer(value, positions, allowed)[0]
                h = checkpoint(run, h, use_reentrant=False)
            else:
                h, kv = block(h, positions, allowed, None if past is None else past[i])
                if return_cache: caches.append(kv)
        h = self.norm(h)
        return (h, caches) if return_cache else h

    def loss_sum(self, batch, mean, scale, diagnostics=False):
        h = self(batch['x'], batch['positions'], batch['bos'], batch['allowed'])[batch['mask']]
        # Legacy sigma_floor=.05 and ELU+1+.05 are defined in RAW AE coordinates.
        target = batch['target'][batch['mask']].float()*scale + mean
        raw = self.head.nll(h, target).sum()
        stats = {}
        if diagnostics:
            with torch.no_grad():
                logits, _, sigma = self.head.params(h.detach())
                logpi = logits.log_softmax(-1)
                stats = {'pi_entropy': float(-(logpi.exp()*logpi).sum(-1).mean()),
                         'sigma_min': float(sigma.min()), 'sigma_mean': float(sigma.mean()),
                         'sigma_near_floor_fraction': float((sigma < self.cfg['sigma_floor']*1.1).float().mean())}
        return raw, int(batch['mask'].sum()), stats


def history_batch(histories, device):
    """BOS + each complete history; final hidden state predicts its next unit."""
    lengths = [len(h)+1 for h in histories]; n = max(lengths); b = len(histories)
    x = torch.zeros(b, n, 48, device=device)
    bos = torch.zeros(b, n, dtype=torch.bool, device=device); bos[:,0] = True
    pos = torch.arange(n, device=device)[None].expand(b,-1)
    valid = torch.arange(n, device=device)[None] < torch.tensor(lengths, device=device)[:,None]
    for i, history in enumerate(histories):
        x[i,1:lengths[i]] = torch.as_tensor(np.asarray(history), device=device, dtype=torch.float32)
    allow = valid[:,:,None] & valid[:,None,:] & (pos[:,None,:] <= pos[:,:,None])
    allow |= (~valid)[:,:,None] & torch.eye(n, device=device, dtype=torch.bool)[None]
    return dict(x=x, positions=pos, bos=bos, allowed=allow[:,None]), lengths


@torch.inference_mode()
def readout(model, hidden, generator, kind='modes_peak', temperature=1.0):
    """Legacy density peaks, untruncated T=1 sampling; deterministic MAP separately."""
    if not torch.isfinite(hidden).all(): raise FloatingPointError('Nonfinite AR hidden state')
    if kind == 'mode':
        result = model.head.mode(hidden)
    elif kind == 'modes_peak':
        values = model.head.modes(hidden)
        if hidden.shape[0] == 1: values = [values]
        chosen = []
        for modes, _, log_density in values:
            if not torch.isfinite(log_density).all() or not torch.isfinite(modes).all():
                raise FloatingPointError('Nonfinite GMM modes')
            index = torch.multinomial((log_density/temperature).softmax(-1), 1, generator=generator)
            chosen.append(modes[index[0]])
        result = torch.stack(chosen)
    else:
        raise ValueError('Use density modes_peak or mode readout')
    if not torch.isfinite(result).all(): raise FloatingPointError('Nonfinite GMM readout')
    return result


@torch.inference_mode()
def generate(model, prefix, units, normalizer, codec, seed=42, history='continuous', kind='modes_peak', temperature=1.0):
    """Return standardized generated latents, decoded bytes, and selected history."""
    device = next(model.parameters()).device
    mean = torch.tensor(normalizer['mean'], device=device); scale = normalizer['scale']
    packed, lengths = history_batch([prefix], device)
    hidden, kv = model(**packed, return_cache=True); hidden = hidden[:,-1]
    rng = torch.Generator(device=device).manual_seed(seed)
    outputs, byte_units = [], []
    encoded = {}
    for j in range(units):
        raw = readout(model, hidden, rng, kind=kind, temperature=temperature)
        z = (raw-mean)/scale
        with torch.autocast(device_type=device.type, enabled=False):
            part = codec.decode(raw)[0]
        outputs.append(z[0]); byte_units.append(part)
        committed = z
        if history == 'reencode' and part:
            if part not in encoded:
                with torch.autocast(device_type=device.type, enabled=False):
                    encoded[part] = torch.as_tensor(codec.encode([part]), device=device)
            committed = (encoded[part]-mean)/scale
        elif history not in ('continuous', 'reencode'):
            raise ValueError('Unknown history policy')
        if j+1 < units:
            p = torch.tensor([[lengths[0]+j]], device=device)
            allow = torch.ones(1,1,1,lengths[0]+j+1, dtype=torch.bool, device=device)
            hidden, kv = model(committed[:,None], p, torch.zeros_like(p, dtype=torch.bool),
                               allow, past=kv, return_cache=True)
            hidden = hidden[:,0]
    return torch.stack(outputs), byte_units
