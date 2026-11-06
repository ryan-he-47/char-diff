"""FP32 reconstruction scores for UTF-8 AE path candidates."""
from __future__ import annotations

from collections.abc import Sequence
import math

import torch

from .ae_paths import CandidateScore

EOS_ID = 256
IGNORE_INDEX = -100


def tensorize(units: Sequence[bytes], l_max: int, device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return byte_ids, byte lengths and padding mask for nonempty UTF-8 units.

    Lengths and padding are measured in bytes; endpoint validity is measured
    by strict UTF-8 decoding. NUL remains an ordinary byte, independent of mask.
    """
    if not units:
        raise ValueError("tensorize requires at least one unit")
    if isinstance(l_max, bool) or not isinstance(l_max, int) or l_max < 1:
        raise ValueError("l_max must be a positive integer")
    ids = torch.zeros((len(units), l_max), dtype=torch.long)
    lengths = []
    for i, unit in enumerate(units):
        if not isinstance(unit, bytes):
            raise TypeError("AE units must be bytes")
        if not 1 <= len(unit) <= l_max:
            raise ValueError(f"AE units must have 1..{l_max} bytes; got {len(unit)}")
        unit.decode("utf-8", errors="strict")
        lengths.append(len(unit))
        ids[i, :len(unit)] = torch.tensor(list(unit), dtype=torch.long)
    lens = torch.tensor(lengths, dtype=torch.long)
    pad_mask = torch.arange(l_max)[None] >= lens[:, None]
    return ids.to(device), lens.to(device), pad_mask.to(device)


def targets_from_lengths(byte_ids: torch.Tensor, lens: torch.Tensor) -> torch.Tensor:
    """Build (B, l_max+1) targets: bytes, explicit EOS, then ignore_index=-100.

    Inputs follow tensorize's contract. This helper avoids GPU-to-CPU checks
    in the training loss path; each sequence has exactly lens+1 valid targets.
    """
    if byte_ids.ndim != 2 or lens.ndim != 1 or len(lens) != len(byte_ids):
        raise ValueError("Expected byte_ids (B,L) and lens (B)")
    extra = torch.zeros_like(byte_ids[:, :1])
    targets = torch.cat((byte_ids, extra), dim=1)
    positions = torch.arange(targets.shape[1], device=byte_ids.device)[None]
    targets = targets.masked_fill(positions == lens[:, None], EOS_ID)
    return targets.masked_fill(positions > lens[:, None], IGNORE_INDEX)


class AEQualityScorer:
    """Batch scorer for ae_paths; quality_kind is 'nll' or 'entropy' (nats).

    The supplied model is moved to device and FP32. Scoring temporarily selects
    eval mode and restores its previous train/eval flag. No model weights are
    loaded here and gradients/requires_grad flags are left untouched.
    """
    def __init__(self, ae, *, quality_kind: str = "nll", device=None, batch_size: int = 512):
        if quality_kind not in ("nll", "entropy"):
            raise ValueError("quality_kind must be 'nll' or 'entropy'")
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        if device is None:
            parameter = next(ae.parameters(), None)
            device = parameter.device if parameter is not None else "cpu"
        self.device = torch.device(device)
        self.ae = ae.to(device=self.device, dtype=torch.float32)
        self.l_max = int(ae.l_max)
        self.quality_kind = quality_kind
        self.batch_size = batch_size

    def __call__(self, units: Sequence[bytes]) -> list[CandidateScore]:
        records = self.score_records(units)
        name = "target_nll_nats" if self.quality_kind == "nll" else "entropy_nats"
        return [CandidateScore(row["feasible"], row[name]) for row in records]

    @torch.inference_mode()
    def score_records(self, units: Sequence[bytes]) -> list[dict]:
        """One model forward per batch; all aggregates include the true EOS slot."""
        records = []
        was_training = self.ae.training
        self.ae.eval()
        try:
            with torch.autocast(device_type=self.device.type, enabled=False):
                for start in range(0, len(units), self.batch_size):
                    part = units[start:start + self.batch_size]
                    ids, lens, mask = tensorize(part, self.l_max, self.device)
                    targets = targets_from_lengths(ids, lens)
                    valid = targets != IGNORE_INDEX
                    safe_targets = targets.clamp_min(0)
                    logits = self.ae(ids, mask).float()
                    if logits.shape != (len(part), self.l_max + 1, 257):
                        raise ValueError("AE logits must have shape (B,l_max+1,257)")
                    if not torch.isfinite(logits).all():
                        raise FloatingPointError("Non-finite AE candidate logits")
                    logp = logits.log_softmax(-1)
                    target_logp = logp.gather(-1, safe_targets[..., None]).squeeze(-1)
                    nll = (-target_logp).masked_fill(~valid, 0).sum(-1)
                    entropy_by_position = -(logp.exp() * logp).sum(-1)
                    entropy = entropy_by_position.masked_fill(~valid, 0).sum(-1)
                    pred = logits.argmax(-1)
                    feasible = ((pred == targets) | ~valid).all(-1)
                    true_logits = logits.gather(-1, safe_targets[..., None]).squeeze(-1)
                    other_logits = logits.scatter(-1, safe_targets[..., None], -math.inf).amax(-1)
                    margins = (true_logits - other_logits).masked_fill(~valid, math.inf).amin(-1)
                    row_ids = torch.arange(len(part), device=self.device)
                    eos_argmax = pred[row_ids, lens] == EOS_ID
                    eos_probability = logp[row_ids, lens, EOS_ID].exp()
                    values = torch.stack((nll, entropy, margins, eos_probability), dim=1).cpu().tolist()
                    exact = feasible.cpu().tolist()
                    eos_ok = eos_argmax.cpu().tolist()
                    for unit, ok, eos, (n, h, margin, eos_p) in zip(part, exact, eos_ok, values):
                        records.append({
                            "unit_hex": unit.hex(), "bytes": len(unit), "feasible": ok,
                            "eos_argmax": eos, "eos_probability": eos_p,
                            "target_nll_nats": n, "entropy_nats": h,
                            "mean_entropy_nats": h / (len(unit) + 1),
                            "min_logit_margin": margin,
                        })
        finally:
            self.ae.train(was_training)
        return records
