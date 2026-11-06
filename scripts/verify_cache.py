"""Portable cache admission and optional frozen-decoder reconstruction audit."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from char_diff.codec import Codec
from char_diff.data import LatentCache
from char_diff.utils import ROOT, write_json


def verify(path, codec_units=0, device="cpu"):
    cache = LatentCache(path, verify_hashes=True)
    m = cache.manifest
    n = m["unique_units"]
    if not np.isfinite(cache.vectors).all() or cache.scale <= 0:
        raise ValueError("Invalid latent table or normalizer")
    counts = np.fromfile(cache.root / "train_counts.u64", dtype="<u8")
    if len(counts) != n or counts.sum() != cache.units:
        raise ValueError("Training frequency table mismatch")
    mean = (cache.vectors.astype(np.float64) * counts[:, None]).sum(0) / cache.units
    variance = (cache.vectors.astype(np.float64)**2 * counts[:, None]).sum() / (48 * cache.units)
    if abs(mean).max() > 1e-5 or abs(variance - 1) > 1e-5:
        raise ValueError("Frequency-weighted standardization mismatch")
    checked = {}
    for split in ("train", "dev", "test"):
        c = cache if split == "train" else LatentCache(path, split)
        meta = m["splits"][split]
        if len(c.offsets) != meta["stories"] + 1 or c.offsets[0] != 0 or c.offsets[-1] != c.units:
            raise ValueError(f"{split}: story offsets mismatch")
        if np.any(c.offsets[1:] <= c.offsets[:-1]):
            raise ValueError(f"{split}: empty or reversed story range")
        if len(c.ids) != c.units or len(c) != meta["windows"] or len(c.valid) != len(c):
            raise ValueError(f"{split}: array lengths mismatch")
        if c.windows[0] != 0 or c.windows[-1] != len(c.segments) or np.any(c.windows[1:] <= c.windows[:-1]):
            raise ValueError(f"{split}: packed window offsets mismatch")
        actual_counts = np.zeros(n, dtype=np.uint64)
        for start in range(0, c.units, 4_000_000):
            chunk = np.asarray(c.ids[start:start + 4_000_000])
            if len(chunk) and chunk.max() >= n:
                raise ValueError(f"{split}: out-of-range unit ID")
            if split == "train":
                actual_counts += np.bincount(chunk.astype(np.int64), minlength=n).astype(np.uint64)
        if split == "train" and not np.array_equal(actual_counts, counts):
            raise ValueError("Training frequencies differ from occurrence IDs")
        if len(c.segments):
            # Sorting by story/start proves every target appears once, including long stories.
            order = np.lexsort((c.segments[:, 1], c.segments[:, 0]))
            doc, start, end, prefix = c.segments[order].T
            if doc.min() != 0 or doc.max() != meta["stories"] - 1:
                raise ValueError(f"{split}: story coverage mismatch")
            first = np.r_[True, doc[1:] != doc[:-1]]
            last = np.r_[doc[1:] != doc[:-1], True]
            lengths = np.diff(c.offsets).astype(np.int64)
            if (not np.array_equal(doc[first], np.arange(meta["stories"]))
                    or np.any(start[first] != 0)
                    or np.any(start[1:][~first[1:]] != end[:-1][~first[1:]])
                    or np.any(end[last] != lengths[doc[last]])
                    or np.any(end <= start)
                    or np.any(end - start > m["layout"]["window"])
                    or np.any(prefix != np.maximum(0, start - m["layout"]["carry"]))):
                raise ValueError(f"{split}: target/prefix coverage mismatch")
            targets = c.segments[:, 2] - c.segments[:, 1]
            per_window = np.add.reduceat(targets, np.asarray(c.windows[:-1], dtype=np.int64))
            if not np.array_equal(per_window, c.valid):
                raise ValueError(f"{split}: per-window target counts mismatch")
        checked[split] = meta
        print(f"PASS {split}: hashes, IDs, offsets, exact target coverage", flush=True)
    offsets = np.fromfile(cache.root / "unit_offsets.u64", dtype="<u8")
    raw = (cache.root / "units.bin").read_bytes()
    if len(offsets) != n + 1 or offsets[0] != 0 or offsets[-1] != len(raw):
        raise ValueError("Unit byte offsets mismatch")
    lengths = np.diff(offsets)
    if np.any(lengths == 0) or np.any(lengths > 32):
        raise ValueError("Unit length outside frozen codec range")
    report = {"cache_fingerprint": cache.fingerprint, "files_verified": len(m["files"]),
              "bytes": sum(x["bytes"] for x in m["files"].values()), "splits": checked,
              "weighted_mean_max_abs": float(abs(mean).max()), "weighted_variance": float(variance)}
    if codec_units:
        codec = Codec(device=device)
        if codec.fingerprint != m["codec"]:
            raise ValueError("Cache codec fingerprint mismatch")
        selected = np.arange(n) if codec_units < 0 or codec_units >= n else np.sort(
            np.random.default_rng(31415).choice(n, codec_units, replace=False))
        vectors = np.fromfile(cache.root / "vectors.f32", dtype="<f4").reshape(n, 48)
        mismatches = []; correct = 0; weighted_correct = 0; weighted_total = 0
        for start in range(0, len(selected), 512):
            indices = selected[start:start + 512]
            outputs = codec.decode(vectors[indices])
            for ident, output in zip(indices, outputs):
                target = raw[int(offsets[ident]):int(offsets[ident + 1])]
                ok = output == target; correct += int(ok)
                weighted_total += int(counts[ident]); weighted_correct += int(counts[ident]) * ok
                if not ok and len(mismatches) < 32:
                    mismatches.append({"id": int(ident), "expected_hex": target.hex(), "decoded_hex": output.hex(),
                                       "train_occurrences": int(counts[ident])})
            if start % 8192 == 0:print(f"codec: {min(start + 512, len(selected))}/{len(selected)} units", flush=True)
        report["codec_audit"] = {"scope": "all_unique" if len(selected) == n else "sampled_unique",
                                  "device": device, "precision": "fp32", "unique_checked": len(selected),
                                  "unique_exact": correct, "selected_train_occurrences": weighted_total,
                                  "selected_train_exact": weighted_correct, "mismatch_examples": mismatches}
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("cache", nargs="?", default="data/tinystories")
    p.add_argument("--codec-units", type=int, default=0, help="0: structural checks; -1: decode every unique unit")
    p.add_argument("--device", default="cpu")
    p.add_argument("--output", default="runs/cache-verification.json")
    args = p.parse_args(); torch.set_num_threads(4)
    result = verify(ROOT / args.cache, args.codec_units, args.device)
    write_json(args.output, result)
    print(json.dumps(result, ensure_ascii=True), flush=True)
