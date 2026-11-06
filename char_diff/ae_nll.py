"""Compact FP32 byte+EOS scoring for the NLL-only segmentation path."""
from __future__ import annotations

import numpy as np
import torch

from .ae_paths import CandidateScore
from .ae_quality import AEQualityScorer, targets_from_lengths, IGNORE_INDEX


def tensorize_compact(units, l_max, device):
    """Build one contiguous byte buffer instead of a tensor per candidate."""
    if not units:
        raise ValueError("Expected nonempty candidate batch")
    lengths = []
    for unit in units:
        if not isinstance(unit, bytes) or not 1 <= len(unit) <= l_max:
            raise ValueError("Candidates must be nonempty bytes within l_max")
        unit.decode("utf-8", errors="strict")
        lengths.append(len(unit))
    packed = b"".join(unit.ljust(l_max, b"\0") for unit in units)
    values = np.frombuffer(packed, dtype=np.uint8).reshape(len(units), l_max).copy()
    ids = torch.from_numpy(values).to(device=device, dtype=torch.long)
    lens = torch.tensor(lengths, dtype=torch.long, device=device)
    mask = torch.arange(l_max, device=device)[None] >= lens[:, None]
    return ids, lens, mask


class AENLLScorer(AEQualityScorer):
    """Same full AE forward and NLL reductions; omit diagnostic-only outputs.

    Full decoder queries preserve the numerical path of AEQualityScorer.
    Candidate order and batch size remain part of cache provenance.
    """

    def __init__(self, ae, *, device=None, batch_size=512, precision="fp32"):
        if precision not in ("fp32", "fp16"):
            raise ValueError("Expected fp32 or fp16 scoring")
        super().__init__(ae, quality_kind="nll", device=device, batch_size=batch_size)
        if precision == "fp16" and self.device.type != "cuda":
            raise ValueError("FP16 scoring probe requires CUDA")
        self.precision = precision

    @torch.inference_mode()
    def __call__(self, units):
        output = []
        was_training = self.ae.training
        self.ae.eval()
        try:
            with torch.autocast(device_type=self.device.type, dtype=torch.float16,
                                enabled=self.precision == "fp16"):
                for start in range(0, len(units), self.batch_size):
                    part = units[start:start + self.batch_size]
                    ids, lens, mask = tensorize_compact(part, self.l_max, self.device)
                    targets = targets_from_lengths(ids, lens)
                    valid = targets != IGNORE_INDEX
                    logits = self.ae(ids, mask).float()
                    if logits.shape != (len(part), self.l_max + 1, 257):
                        raise ValueError("AE logits must have shape (B,l_max+1,257)")
                    if not torch.isfinite(logits).all():
                        raise FloatingPointError("Non-finite AE candidate logits")
                    logp = logits.log_softmax(-1)
                    nll = -logp.gather(-1, targets.clamp_min(0)[..., None]).squeeze(-1)
                    nll = nll.masked_fill(~valid, 0).sum(-1)
                    feasible = ((logits.argmax(-1) == targets) | ~valid).all(-1)
                    values = torch.stack((feasible.float(), nll), dim=-1).cpu().tolist()
                    output.extend(CandidateScore(bool(ok), q) for ok, q in values)
        finally:
            self.ae.train(was_training)
        return output
