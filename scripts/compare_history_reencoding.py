"""Paired inference experiment: continuous generated history versus AE reencoding."""
import argparse
import json
import time
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from char_diff.codec import Codec, HistoryReencoder
from char_diff.data import LatentCache
from char_diff.flow import generate
from char_diff.model import Denoiser
from char_diff.utils import ROOT, sha256, write_json


def summarize(records):
    groups = {}
    for name, subset in [("primary", [r for r in records if r["units"] == 128]),
                         ("long", [r for r in records if r["units"] == 256])]:
        arms = {}
        for mode in ["continuous", "reencoded"]:
            values = [r["arms"][mode] for r in subset]
            total_units = sum(r["units"] for r in subset)
            matched = sum(v["seen_training_units"] for v in values)
            arms[mode] = {"cases": len(values), "units": total_units,
                          "seen_training_units": matched, "seen_training_unit_rate": matched / total_units,
                          "valid_utf8": sum(v["utf8_valid"] for v in values),
                          "empty_units": sum(v["empty_units"] for v in values),
                          "mean_seconds": float(np.mean([v["seconds"] for v in values]))}
        differences = [r["arms"]["reencoded"]["seen_training_units"] - r["arms"]["continuous"]["seen_training_units"] for r in subset]
        groups[name] = {"arms": arms, "seen_unit_count_delta": {
            "higher": sum(x > 0 for x in differences), "same": sum(x == 0 for x in differences),
            "lower": sum(x < 0 for x in differences)},
            "mean_trajectory_rmse_per_block": np.mean([r["trajectory_rmse_per_block"] for r in subset], axis=0).tolist()}
    primary = [r for r in records if r["units"] == 128]
    doc_deltas = {}
    for r in primary:
        delta = (r["arms"]["reencoded"]["seen_training_units"] - r["arms"]["continuous"]["seen_training_units"]) / r["units"]
        doc_deltas.setdefault(r["doc"], []).append(delta)
    per_doc = np.array([np.mean(v) for v in doc_deltas.values()])
    indices = np.random.default_rng(20261101).integers(len(per_doc), size=(10000, len(per_doc)))
    boot = per_doc[indices].mean(1)
    stats = [r["arms"]["reencoded"]["history_stats"] for r in records]
    return {"groups": groups, "control_matches_prior": sum(r["control_matches_prior"] for r in records),
            "first_block_exact_pairs": sum(r["first_block_exact"] for r in records),
            "pairs": len(records), "primary_seen_rate_delta": float(per_doc.mean()),
            "primary_seen_rate_delta_story_bootstrap_95ci": np.quantile(boot, [.025, .975]).tolist(),
            "reencoding_totals": {key: sum(v[key] for v in stats) for key in [
                "calls", "units", "projected_units", "fallback_empty", "fallback_oversized", "byte_roundtrip_exact"]}}


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("checkpoint", nargs="?", default="runs/tinystories_x0_22m/last.pt")
    p.add_argument("--out", default="runs/reencode_history_715121")
    args = p.parse_args(); torch.set_num_threads(4)
    out = ROOT / args.out; out.mkdir(parents=True, exist_ok=True)
    if (out / "pairs.jsonl").exists():raise FileExistsError("Choose a new experiment directory")
    state = torch.load(ROOT / args.checkpoint, map_location="cpu", weights_only=False)
    selected = json.loads((ROOT / "docs/experiments/baseline/evidence/main-run-715121-prompts.json").read_text(encoding="utf-8"))
    cache = LatentCache(ROOT / state["config"]["data"]["cache"], "dev")
    if state["cache_fingerprint"] != cache.fingerprint or selected["cache_fingerprint"] != cache.fingerprint:
        raise ValueError("Checkpoint, prompts and cache differ")
    codec = Codec(device="cuda")
    if codec.fingerprint != state["codec"]:raise ValueError("Codec differs")
    model = Denoiser(state["config"]["model"]).cuda().eval(); model.load_state_dict(state["ema"])
    mean = torch.tensor(state["normalizer"]["mean"], dtype=torch.float32, device="cuda")
    scale = state["normalizer"]["scale"]
    offsets = np.fromfile(cache.root / "unit_offsets.u64", dtype="<u8")
    blob = (cache.root / "units.bin").read_bytes()
    counts = np.fromfile(cache.root / "train_counts.u64", dtype="<u8")
    seen = {blob[int(offsets[i]):int(offsets[i + 1])] for i in np.flatnonzero(counts)}
    previous = [json.loads(line) for line in (ROOT / "docs/experiments/baseline/evidence/main-run-715121-samples.jsonl").read_text(encoding="utf-8").splitlines()]
    prior = {(r["doc"], r["seed"], r["units"]): r for r in previous
             if r["stage"] == "epoch_2.000" and r["weights"] == "ema" and r["doc"] is not None}
    jobs = [(q, seed, 128) for q in selected["prompts"] for seed in [42, 123]]
    jobs.extend((q, 42, 256) for q in selected["prompts"][:2])
    protocol = {"date": "2026-11-01", "checkpoint": str((ROOT / args.checkpoint).resolve()),
                "checkpoint_sha256": sha256(ROOT / args.checkpoint), "checkpoint_step": state["state"]["step"],
                "weights": "ema", "cache_fingerprint": cache.fingerprint, "codec": codec.fingerprint,
                "model_precision": "fp16", "codec_precision": "fp32", "device": torch.cuda.get_device_name(),
                "torch": torch.__version__, "steps_per_block": 32, "block": 16, "prefix_units": 32,
                "primary_pairs": 32, "long_pairs": 2, "output": "raw denoised blocks; intervention changes future history only"}
    write_json(out / "protocol.json", protocol)
    records = []; latents = {}; started = time.perf_counter()
    with open(out / "pairs.jsonl", "x", encoding="utf-8", buffering=1) as journal:
        for case, (prompt, seed, units) in enumerate(jobs, 1):
            doc = prompt["doc"]; prefix = cache.story(doc)[:32]
            arms = {}; outputs = {}; decoded = {}; transform = None
            for mode in ["continuous", "reencoded"]:
                transform = HistoryReencoder(codec, state["normalizer"], keep_trace=True) if mode == "reencoded" else None
                tick = time.perf_counter()
                with torch.autocast("cuda", dtype=torch.float16):
                    z = generate(model, prefix, units, steps=32, seed=seed, history_transform=transform)
                if not torch.isfinite(z).all():raise FloatingPointError("Non-finite output")
                pieces = codec.decode(z * scale + mean); raw = b"".join(pieces)
                try:
                    text = raw.decode("utf-8"); valid_utf8 = True
                except UnicodeDecodeError:
                    text = raw.decode("utf-8", errors="replace"); valid_utf8 = False
                arms[mode] = {"generated": text, "generated_hex": raw.hex(), "bytes": len(raw),
                              "unit_lengths": [len(v) for v in pieces], "utf8_valid": valid_utf8,
                              "seen_training_units": sum(v in seen for v in pieces),
                              "empty_units": sum(not v for v in pieces), "seconds": time.perf_counter() - tick}
                if transform:
                    arms[mode]["history_stats"] = transform.summary()
                    arms[mode]["history_readout_equals_emission"] = transform.decoded_units == pieces
                    latents[f"case_{case}_history"] = np.concatenate(transform.blocks)
                outputs[mode] = z.cpu().numpy(); decoded[mode] = pieces
                latents[f"case_{case}_{mode}"] = outputs[mode]
            first_block_exact = np.array_equal(outputs["continuous"][:16], outputs["reencoded"][:16]) and decoded["continuous"][:16] == decoded["reencoded"][:16]
            matches_prior = arms["continuous"]["generated_hex"] == prior[(doc, seed, units)]["generated_hex"]
            if not first_block_exact or not matches_prior:
                raise AssertionError("Paired control/first-block validation failed")
            trajectory = [float(np.sqrt(np.mean((outputs["continuous"][i:i+16]-outputs["reencoded"][i:i+16])**2))) for i in range(0, units, 16)]
            record = {"case": case, "doc": doc, "seed": seed, "units": units, "prompt": prompt["prompt"],
                      "reference": prompt["reference"], "control_matches_prior": matches_prior,
                      "first_block_exact": first_block_exact, "trajectory_rmse_per_block": trajectory, "arms": arms}
            records.append(record); journal.write(json.dumps(record, ensure_ascii=False) + "\n")
            print(json.dumps({"pair": case, "total": len(jobs), "doc": doc, "seed": seed, "units": units,
                              "control_exact": matches_prior, "first_block_exact": first_block_exact,
                              "seen_units": {k:v["seen_training_units"] for k,v in arms.items()},
                              "history": arms["reencoded"]["history_stats"]}), flush=True)
    np.savez(out / "latents.npz", **latents)
    summary = {"protocol": protocol, **summarize(records), "elapsed_seconds": time.perf_counter()-started}
    write_json(out / "summary.json", summary)
    lines = ["# 生成历史重编码：逐对原始输出", "", "固定前缀、种子、模型与去噪预算；每对第一块完全相同。", ""]
    for r in records:
        lines.extend([f"## {r['case']}. doc={r['doc']} / seed={r['seed']} / units={r['units']}", "", "**前缀**", "", r["prompt"], ""])
        for mode, label in [("continuous", "连续历史"), ("reencoded", "读出后重编码历史")]:
            lines.extend([f"**{label}**", "", r["arms"][mode]["generated"], ""])
    (out / "pairs.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":main()
