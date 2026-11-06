"""Slot preservation, fallback handling and the frozen codec's FP32 boundary."""
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from char_diff.codec import Codec, HistoryReencoder
from char_diff.utils import ROOT, write_json


def main():
    torch.set_num_threads(2)
    class FakeCodec:
        device = "cpu"
        ae = SimpleNamespace(l_max=32)
        def decode(self, z):
            assert not torch.is_autocast_enabled("cpu") and z.dtype == torch.float32
            return [b"a", b"", b"x"*33, b"b"] if len(z) == 4 else [b"a", b"b"]
        def encode(self, units):
            assert not torch.is_autocast_enabled("cpu")
            return np.full((len(units), 48), 3., np.float32)
    normalizer = {"mean": [0.5]*48, "scale": 2.}
    projection = HistoryReencoder(FakeCodec(), normalizer, keep_trace=True)
    original = torch.zeros(4, 48)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        result = projection(original)
    torch.testing.assert_close(original, torch.zeros_like(original), atol=0, rtol=0)
    torch.testing.assert_close(result[[0, 3]], torch.full((2, 48), 1.25), atol=0, rtol=0)
    torch.testing.assert_close(result[[1, 2]], original[[1, 2]], atol=0, rtol=0)
    stats = projection.summary()
    assert stats["projected_units"] == 2 and stats["fallback_empty"] == 1 and stats["fallback_oversized"] == 1
    assert stats["byte_roundtrip_exact"] == 2 and projection.blocks[0].shape == (4, 48)
    print("PASS normalization, slot-preserving fallbacks and non-mutating codec projection", flush=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    real = Codec(device=device)
    offsets = np.fromfile(ROOT / "data/tiny_smoke/unit_offsets.u64", dtype="<u8")
    blob = (ROOT / "data/tiny_smoke/units.bin").read_bytes()
    units = [blob[int(offsets[i]):int(offsets[i+1])] for i in range(4)]
    calls = []
    class SpyCodec:
        def __init__(self):self.device=real.device;self.ae=real.ae
        def encode(self, values):
            calls.append(not torch.is_autocast_enabled(device))
            return real.encode(values)
        def decode(self, values):
            calls.append(not torch.is_autocast_enabled(device) and values.dtype==torch.float32)
            return real.decode(values)
    z = torch.tensor((real.encode(units)-0.5)/2, device=device)
    fp32_projection = HistoryReencoder(SpyCodec(), normalizer)
    with torch.autocast(device, dtype=torch.float16 if device=="cuda" else torch.bfloat16):
        rebuilt = fp32_projection(z)
    assert all(calls) and rebuilt.dtype==torch.float32 and torch.isfinite(rebuilt).all()
    assert fp32_projection.summary()["byte_roundtrip_exact"]==4
    assert all(not p.requires_grad for p in real.ae.parameters())
    print(f"PASS real frozen codec stays FP32 inside {device} autocast", flush=True)
    write_json(ROOT / "docs/experiments/history-reencoding/evidence/history-reencoding-tests.json", {
        "passed": ["slot_preserving_fallbacks", "normalization", "nonmutating_projection", "FP32_inside_autocast", "frozen_codec"],
        "device":device,"torch":torch.__version__})


if __name__=="__main__":main()
