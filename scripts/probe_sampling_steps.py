"""Compare the planned 16/32/64-step samplers on identical dev prefixes/seeds."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from char_diff.codec import Codec
from char_diff.data import LatentCache
from char_diff.flow import generate
from char_diff.model import Denoiser
from char_diff.utils import ROOT, write_json


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("checkpoint")
    p.add_argument("--prompts", default="runs/review_715121/panel/prompts.json")
    p.add_argument("--out", default="runs/review_715121/sampling_steps")
    p.add_argument("--count", type=int, default=4)
    args = p.parse_args(); torch.set_num_threads(4)
    out = ROOT / args.out; out.mkdir(parents=True, exist_ok=True)
    if (out / "samples.json").exists():raise FileExistsError("Choose a new output directory")
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cache = LatentCache(ROOT / state["config"]["data"]["cache"], "dev")
    selected = json.loads((ROOT / args.prompts).read_text(encoding="utf-8"))
    if cache.fingerprint != state["cache_fingerprint"] or cache.fingerprint != selected["cache_fingerprint"]:
        raise ValueError("Sampling inputs differ")
    codec = Codec(device="cuda")
    if codec.fingerprint != state["codec"]:raise ValueError("Codec differs")
    model = Denoiser(state["config"]["model"]).cuda().eval(); model.load_state_dict(state["ema"])
    mean = torch.tensor(state["normalizer"]["mean"], device="cuda", dtype=torch.float32)
    scale = state["normalizer"]["scale"]
    offsets = np.fromfile(cache.root / "unit_offsets.u64", dtype="<u8")
    blob = (cache.root / "units.bin").read_bytes()
    counts = np.fromfile(cache.root / "train_counts.u64", dtype="<u8")
    seen = {blob[int(offsets[i]):int(offsets[i + 1])] for i in np.flatnonzero(counts)}
    samples = []; vectors = {}; differences = []
    for prompt in selected["prompts"][:args.count]:
        doc = prompt["doc"]; prefix = cache.story(doc)[:32]
        for steps in [16, 32, 64]:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                z = generate(model, prefix, 128, steps, 42,prediction_type=state["config"]["train"].get("prediction_type","x0"))
            vectors[f"doc_{doc}_steps_{steps}"] = z.cpu().numpy()
            decoded = codec.decode(z * scale + mean); raw = b"".join(decoded)
            sample = {"doc": doc, "steps": steps, "seed": 42, "prefix": prompt["prompt"],
                      "generated": raw.decode("utf-8", errors="replace"), "generated_hex": raw.hex(),
                      "units": 128, "seen_training_units": sum(v in seen for v in decoded),
                      "empty_units": sum(not v for v in decoded)}
            samples.append(sample)
            write_json(out / "samples.json", samples)
            print(json.dumps({k:sample[k] for k in ["doc", "steps", "seen_training_units", "empty_units"]}), flush=True)
        a = vectors[f"doc_{doc}_steps_32"]; b = vectors[f"doc_{doc}_steps_64"]
        differences.append({"doc": doc, "first_block_32_vs_64_rmse": float(np.sqrt(np.mean((a[:16]-b[:16])**2))),
                            "whole_sample_32_vs_64_rmse": float(np.sqrt(np.mean((a-b)**2)))})
    np.savez(out / "latents.npz", **vectors)
    summary = {"checkpoint_step": state["state"]["step"], "weights": "ema", "precision": "fp16",
               "cache_fingerprint": cache.fingerprint, "cases": args.count, "seed": 42,
               "differences": differences, "by_steps": {
                   str(steps):{"generated_units":sum(s["units"] for s in samples if s["steps"]==steps),
                               "seen_training_units":sum(s["seen_training_units"] for s in samples if s["steps"]==steps)}
                   for steps in [16,32,64]}}
    write_json(out / "summary.json", summary)
    lines = ["# Euler 步数配对预览", "", "同一模型、前缀、种子和 128-unit 预算；原生字节读出。", ""]
    for s in samples:
        lines.extend([f"## doc={s['doc']} / steps={s['steps']}", "", s["prefix"], "", s["generated"], ""])
    (out / "samples.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":main()
