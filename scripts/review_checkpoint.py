"""Fixed dev prompts, paired milestone previews, and denoising diagnostics."""
import argparse
import json
import time
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from char_diff.codec import Codec
from char_diff.data import LatentCache
from char_diff.flow import corrupt, generate, loss_sum, prediction_to_data
from char_diff.model import Denoiser, CLEAN, NOISY
from char_diff.utils import ROOT, sha256, write_json


@torch.inference_mode()
def denoising(model, cache, windows, device, history=True,prediction_type="x0"):
    rows = []
    for t in [0.05, 0.25, 0.5, 0.75, 0.95]:
        total = 0.; count = 0
        rng = torch.Generator(device=device).manual_seed(31415)
        for i in range(min(windows, len(cache))):
            batch = cache.batch([i], device)
            x, times = corrupt(batch, rng, t)
            allowed = batch["allowed"]
            if not history:
                # Retain BOS and the current noisy block, remove clean-history keys.
                remove = (batch["roles"] == NOISY)[:, None, :, None] & (batch["roles"] == CLEAN)[:, None, None, :]
                allowed = allowed & ~remove
            pred = model(x, times, batch["roles"], batch["positions"], batch["lengths"], allowed)
            value, n = loss_sum(prediction_to_data(pred,x,times,prediction_type), batch)
            total += float(value); count += n
        row = {"time": t, "mse": total / (48 * count), "units": count}
        rows.append(row)
        print(json.dumps({"evaluation": "full_history" if history else "BOS_only", **row}), flush=True)
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("checkpoint")
    p.add_argument("--compare", action="append", default=[])
    p.add_argument("--out", default="runs/review_715121/panel")
    p.add_argument("--prompts", type=int, default=16)
    p.add_argument("--seeds", default="42,123")
    p.add_argument("--eval-windows", type=int, default=64)
    p.add_argument("--device", default="cuda")
    p.add_argument("--precision", choices=["fp32", "fp16", "bf16"], default="fp16")
    args = p.parse_args()
    if args.prompts < 1 or args.eval_windows < 1:
        p.error("Prompt and evaluation counts must be positive")
    out = ROOT / args.out; out.mkdir(parents=True, exist_ok=True)
    if (out / "samples.jsonl").exists():
        raise FileExistsError("Choose a new review output directory")
    torch.set_num_threads(4)
    device = torch.device(args.device)
    dtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[args.precision]
    seeds = [int(v) for v in args.seeds.split(",")]
    metadata = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cache = LatentCache(ROOT / metadata["config"]["data"]["cache"], "dev")
    if cache.fingerprint != metadata["cache_fingerprint"]:
        raise ValueError("Review cache differs from the checkpoint")
    normalizer = metadata["normalizer"]; codec_id = metadata["codec"]
    del metadata
    codec = Codec(device=device)
    if codec.fingerprint != codec_id:
        raise ValueError("Frozen codec differs from the checkpoint")
    candidates = np.flatnonzero(np.diff(cache.offsets) >= 64)
    selected = np.random.default_rng(20261101).choice(candidates, args.prompts, replace=False)
    byte_offsets = np.fromfile(cache.root / "unit_offsets.u64", dtype="<u8")
    raw_table = (cache.root / "units.bin").read_bytes()
    def decode_ids(ids):
        return b"".join(raw_table[int(byte_offsets[i]):int(byte_offsets[i + 1])] for i in ids)
    prompts = []
    for ident in selected:
        ident = int(ident); start = int(cache.offsets[ident]); end = int(cache.offsets[ident + 1])
        ids = cache.ids[start:end]
        raw_prompt = decode_ids(ids[:32]); reference = decode_ids(ids[32:160])
        prompts.append({"doc": ident, "prompt": raw_prompt.decode("utf-8", errors="replace"),
                        "prompt_hex": raw_prompt.hex(), "prefix_units": 32, "story_units": len(ids),
                        "reference": reference.decode("utf-8", errors="replace"), "reference_hex": reference.hex()})
    write_json(out / "prompts.json", {"cache_fingerprint": cache.fingerprint, "selection_seed": 20261101,
                                      "selection": "uniform dev stories with at least 64 units", "prompts": prompts})
    scale = normalizer["scale"]
    mean = torch.tensor(normalizer["mean"], dtype=torch.float32, device=device)
    stages = [Path(args.checkpoint)] + [Path(v) for v in args.compare]
    report = {"split": "dev", "cache_fingerprint": cache.fingerprint, "prompt_seed": 20261101,
              "prompt_count": args.prompts, "seeds": seeds, "prefix_units": 32, "sample_precision": args.precision,
              "evaluation_precision": "fp32", "steps_per_block": 32, "stages": []}
    samples = []; started = time.perf_counter()
    journal = open(out / "samples.jsonl", "x", encoding="utf-8", buffering=1)
    try:
        for stage_index, path in enumerate(stages):
            state = torch.load(path, map_location="cpu", weights_only=False)
            if state["cache_fingerprint"] != cache.fingerprint or state["codec"] != codec_id or state["normalizer"] != normalizer:
                raise ValueError("Milestone input representation differs")
            model = Denoiser(state["config"]["model"]).to(device).eval()
            prediction_type=state["config"]["train"].get("prediction_type","x0")
            epoch = state["state"]["seen_units"] / cache.manifest["splits"]["train"]["units"]
            tag = f"epoch_{epoch:.3f}"
            entry = {"checkpoint": str(path.resolve()), "sha256": sha256(path), "epoch": epoch,
                     "step": state["state"]["step"], "weights": {}}
            print(json.dumps({"stage": tag, "step": state["state"]["step"]}), flush=True)
            for weight in (["ema", "model"] if stage_index == 0 else ["ema"]):
                model.load_state_dict(state[weight])
                metrics = {"full_history": denoising(model, cache, args.eval_windows, device,prediction_type=prediction_type)}
                if stage_index == 0 and weight == "ema":
                    metrics["BOS_only"] = denoising(model, cache, args.eval_windows, device, history=False,prediction_type=prediction_type)
                entry["weights"][weight] = metrics
                group = prompts if weight == "ema" and stage_index == 0 else prompts[:8 if weight == "ema" else 4]
                group_seeds = seeds if stage_index == 0 and weight == "ema" else seeds[:1]
                jobs = [(q, seed, 128, "continuation") for q in group for seed in group_seeds]
                if stage_index == 0 and weight == "ema":
                    jobs.extend([(None, seed, 128, "BOS") for seed in seeds])
                    jobs.extend([(q, seeds[0], 256, "long_continuation") for q in prompts[:2]])
                for prompt, seed, units, kind in jobs:
                    prefix = cache.story(prompt["doc"])[:32] if prompt else np.empty((0, 48), np.float32)
                    tick = time.perf_counter()
                    with torch.autocast(device_type=device.type, dtype=dtype,
                                        enabled=device.type == "cuda" and dtype != torch.float32):
                        z = generate(model, prefix, units, 32, seed,prediction_type=prediction_type)
                    if not torch.isfinite(z).all():
                        raise FloatingPointError("Generated latents contain non-finite values")
                    decoded = codec.decode(z * scale + mean)
                    raw = b"".join(decoded)
                    try:
                        text = raw.decode("utf-8"); valid_utf8 = True
                    except UnicodeDecodeError:
                        text = raw.decode("utf-8", errors="replace"); valid_utf8 = False
                    sample = {"stage": tag, "step": state["state"]["step"], "weights": weight, "kind": kind,
                              "doc": prompt["doc"] if prompt else None, "seed": seed,
                              "prompt": prompt["prompt"] if prompt else "", "generated": text,
                              "reference": prompt["reference"] if prompt else None,
                              "generated_hex": raw.hex(), "unit_lengths": [len(v) for v in decoded],
                              "utf8_valid": valid_utf8, "units": units, "bytes": len(raw),
                              "denoiser_calls": (units + 15) // 16 * 32, "seconds": time.perf_counter() - tick}
                    samples.append(sample); journal.write(json.dumps(sample, ensure_ascii=False) + "\n")
                    print(json.dumps({"sample": len(samples), "stage": tag, "weights": weight, "kind": kind,
                                      "doc": sample["doc"], "seed": seed, "utf8_valid": valid_utf8}), flush=True)
            report["stages"].append(entry)
            write_json(out / "metrics.json", report)
            del model, state
            if device.type == "cuda":torch.cuda.empty_cache()
    finally:
        journal.close()
    report["elapsed_seconds"] = time.perf_counter() - started
    report["sample_count"] = len(samples)
    report["valid_utf8_samples"] = sum(s["utf8_valid"] for s in samples)
    write_json(out / "metrics.json", report)
    lines = ["# 固定开发集续写预览", "", "固定 32-unit 前缀；正文为原生字节读出。参考续文用于配对阅读。", ""]
    for i, s in enumerate(samples, 1):
        lines.extend([f"## {i}. {s['stage']} / {s['weights']} / {s['kind']} / doc={s['doc']} / seed={s['seed']}", "",
                      "**前缀**", "", "> " + s["prompt"].replace("\n", "\n> "), "", "**生成**", "", s["generated"], ""])
        if s["reference"]:lines.extend(["**参考续文**", "", s["reference"], ""])
    (out / "samples.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"samples": len(samples), "seconds": report["elapsed_seconds"], "output": str(out)}), flush=True)


if __name__ == "__main__":main()
