"""FP32 AE cycle scoring and one bounded retry before a B1 unit is committed."""
import math
import numpy as np
import torch


class CycleScorer:
    def __init__(self, codec, normalizer):
        self.codec = codec
        self.mean = torch.tensor(normalizer['mean'], dtype=torch.float32, device=codec.device)
        self.scale = float(normalizer['scale'])
        self.encoded = {}

    @torch.inference_mode()
    def __call__(self, x):
        with torch.autocast(device_type=self.codec.device.type, enabled=False):
            x = x.float().reshape(-1, self.mean.numel())
            if not torch.isfinite(x).all(): raise FloatingPointError('Non-finite selection candidate')
            lam = x * self.scale + self.mean
            n = len(x); k = self.codec.ae.l_max + 1
            logits = self.codec.ae.decoder(lam.repeat_interleave(k, 0), torch.arange(k, device=x.device).repeat(n))
            logp = logits.float().log_softmax(-1).reshape(n, k, 257)
            probs = logp.exp(); entropy = -(probs * logp).sum(-1) / np.log(2.)
            best = logp.argmax(-1).cpu().tolist(); raw = []; rows = []
            for i, ids in enumerate(best):
                size = min(ids.index(256) if 256 in ids else k, k - 1)
                value = bytes(ids[:size]); raw.append(value)
                try: value.decode('utf-8'); utf8 = True
                except UnicodeDecodeError: utf8 = False
                eos = ids[size] == 256
                rows.append({'unit_hex': value.hex(), 'bytes': size, 'utf8_valid': utf8,
                             'eos_argmax': eos, 'eos_probability': float(probs[i, size, 256]),
                             'valid': bool(size and utf8 and eos), 'cycle_rmse': None,
                             'mean_entropy_bits': float(entropy[i, :size].mean()) if size else None,
                             'max_entropy_bits': float(entropy[i, :size].max()) if size else None})
            misses = list(dict.fromkeys(v for v in raw if v and v not in self.encoded))
            if misses:
                encoded = torch.as_tensor(self.codec.encode(misses), device=x.device)
                self.encoded.update({v: e for v, e in zip(misses, encoded)})
            for i, value in enumerate(raw):
                if value:
                    center = (self.encoded[value] - self.mean) / self.scale
                    rows[i]['cycle_rmse'] = float((x[i] - center).square().mean().sqrt())
            return rows


def needs_retry(score, threshold):
    return not score['valid'] or score['cycle_rmse'] > threshold


def prefer_retry(first, second):
    """Choose a valid lower-R candidate; ties and two invalid candidates retain the first."""
    return second['valid'] and (not first['valid'] or second['cycle_rmse'] < first['cycle_rmse'])


class CycleRetry:
    def __init__(self, scorer, threshold=.1):
        if not math.isfinite(threshold) or threshold < 0: raise ValueError('Use a finite nonnegative threshold')
        self.scorer = scorer; self.threshold = threshold; self.records = []

    @torch.inference_mode()
    def __call__(self, first, retry):
        original = self.scorer(first)[0]; triggered = needs_retry(original, self.threshold)
        second = None; selected = first; accepted = False
        if triggered:
            candidate = retry(); second = self.scorer(candidate)[0]
            accepted = prefer_retry(original, second)
            if accepted: selected = candidate
        chosen = second if accepted else original
        self.records.append({'unit': len(self.records), 'triggered': triggered, 'accepted_retry': accepted,
                             'changed_bytes': chosen['unit_hex'] != original['unit_hex'],
                             'original': original, 'retry': second, 'selected': chosen})
        return selected

    def summary(self):
        return {'threshold': self.threshold, 'max_retries': 1, 'units': len(self.records),
                'retries': sum(r['triggered'] for r in self.records),
                'accepted_retries': sum(r['accepted_retry'] for r in self.records),
                'changed_units': sum(r['changed_bytes'] for r in self.records),
                'invalid_selected': sum(not r['selected']['valid'] for r in self.records)}
