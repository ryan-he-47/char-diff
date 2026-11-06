from __future__ import annotations
from pathlib import Path
import numpy as np
import torch
from .utils import ROOT, sha256
from .vendor.bltz.config import Cfg
from .vendor.bltz.models.autoencoder import ByteStringAE
from .vendor.bltz.segment_v2 import segment_word
from .vendor.bltz.verify import VerifyResplitter

class Codec:
    def __init__(self, assets=None, device="cpu"):
        self.assets = Path(assets or ROOT / "assets")
        self.device = torch.device(device)
        self.checkpoint = self.assets / "ae_best.pt"
        state = torch.load(self.checkpoint, map_location="cpu", weights_only=False)
        self.ae = ByteStringAE(Cfg(state["cfg"])).to(self.device).eval()
        self.ae.load_state_dict(state["model"], strict=True)
        self.ae.requires_grad_(False)
        self.verifier = VerifyResplitter(self.ae, str(self.assets), device=str(self.device), log_every=0)
        self.fingerprint = {"ae_sha256": sha256(self.checkpoint),
                            "tokenizer_sha256": sha256(self.assets/"tokenizer.json"),
                            "ae_step": state.get("step")}

    def split(self, texts):
        raw = [t.encode("utf-8") if isinstance(t, str) else t for t in texts]
        units = [segment_word(b, str(self.assets), l_max=32, merge="prev") for b in raw]
        out = self.verifier.process_batch(units)
        if any(b"".join(u) != b for u, b in zip(out, raw)):
            raise RuntimeError("Byte conservation check failed")
        return out

    @torch.inference_mode()
    def encode(self, units, batch_size=512):
        results = []
        for start in range(0, len(units), batch_size):
            part = units[start:start+batch_size]
            ids = torch.zeros(len(part), self.ae.l_max, dtype=torch.long, device=self.device)
            lengths = torch.tensor([len(x) for x in part], device=self.device)
            if (lengths > self.ae.l_max).any():
                raise ValueError("Codec input exceeds l_max")
            for i, u in enumerate(part):
                if u:
                    ids[i, :len(u)] = torch.tensor(list(u), device=self.device)
            mask = torch.arange(self.ae.l_max, device=self.device)[None] >= lengths[:, None]
            if (lengths == 0).any():
                raise ValueError("Empty units use control metadata rather than AE encoding")
            results.append(self.ae.encoder(ids, mask).float().cpu().numpy())
        return np.concatenate(results) if results else np.empty((0,48),np.float32)

    @torch.inference_mode()
    def decode(self, latents):
        z = torch.as_tensor(latents, dtype=torch.float32, device=self.device)
        return [bytes(v) for v in self.ae.decode(z)]


class HistoryReencoder:
    """FP32 byte decode/encode projection in standardized coordinates, preserving slots."""
    def __init__(self, codec, normalizer, keep_trace=False):
        self.codec = codec
        self.mean = torch.tensor(normalizer["mean"], dtype=torch.float32, device=codec.device)
        self.scale = float(normalizer["scale"])
        if not np.isfinite(self.scale) or self.scale <= 0:
            raise ValueError("History normalization scale must be positive and finite")
        self.keep_trace = keep_trace
        self.blocks = []; self.decoded_units = []; self.errors = []
        self.calls = self.units = self.projected = self.empty = self.oversized = self.byte_exact = 0

    @torch.inference_mode()
    def __call__(self, block):
        if block.ndim != 2 or block.shape[1] != self.mean.numel():
            raise ValueError("History block differs from the codec dimensions")
        with torch.autocast(device_type=block.device.type, enabled=False):
            source = block.float()
            decoded = self.codec.decode(source * self.scale + self.mean)
            if len(decoded) != len(source):raise ValueError("Codec changed the unit count")
            indices = [i for i, value in enumerate(decoded) if 0 < len(value) <= self.codec.ae.l_max]
            self.calls += 1; self.units += len(source)
            self.empty += sum(not value for value in decoded)
            self.oversized += sum(len(value) > self.codec.ae.l_max for value in decoded)
            result = source.clone()
            if indices:
                values = [decoded[i] for i in indices]
                encoded = self.codec.encode(values)
                vectors = torch.as_tensor(encoded, dtype=torch.float32, device=source.device)
                result[indices] = (vectors - self.mean) / self.scale
                errors = ((result[indices] - source[indices]).square().mean(-1).sqrt()).cpu().tolist()
                self.errors.extend(errors); self.projected += len(indices)
                repeated = self.codec.decode(vectors)
                self.byte_exact += sum(a == b for a, b in zip(values, repeated))
            if not torch.isfinite(result).all():raise FloatingPointError("Non-finite reencoded history")
            if self.keep_trace:
                self.blocks.append(result.cpu().numpy().copy())
                self.decoded_units.extend(decoded)
            return result

    def summary(self):
        return {"calls": self.calls, "units": self.units, "projected_units": self.projected,
                "fallback_empty": self.empty, "fallback_oversized": self.oversized,
                "byte_roundtrip_exact": self.byte_exact, "codec_precision": "fp32",
                "cycle_rmse_mean": float(np.mean(self.errors)) if self.errors else 0.,
                "cycle_rmse_p90": float(np.quantile(self.errors, .9)) if self.errors else 0.}
