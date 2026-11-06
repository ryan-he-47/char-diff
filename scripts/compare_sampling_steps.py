"""Expand the 16/32/64 Euler comparison using validated cached 32-step controls."""
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
from char_diff.flow import generate
from char_diff.model import Denoiser
from char_diff.utils import ROOT, sha256, write_json


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("checkpoint", nargs="?", default="runs/tinystories_x0_22m/last.pt")
    p.add_argument("--out", default="runs/sampling_steps_715121")
    args = p.parse_args(); torch.set_num_threads(4)
    out = ROOT / args.out; out.mkdir(parents=True, exist_ok=True)
    if (out / "samples.jsonl").exists():raise FileExistsError("Choose a new output directory")
    state = torch.load(ROOT / args.checkpoint, map_location="cpu", weights_only=False)
    cached_summary = json.loads((ROOT / "docs/experiments/history-reencoding/evidence/history-reencoding-715121-summary.json").read_text(encoding="utf-8"))
    checkpoint_sha = sha256(ROOT / args.checkpoint)
    if checkpoint_sha != cached_summary["protocol"]["checkpoint_sha256"]:
        raise ValueError("Cached comparison uses another checkpoint")
    # The fresh four-prompt controls below validate bytes and continuous arrays
    # before reusing prior outputs, including across backwards-compatible additions.
    selected = json.loads((ROOT / "docs/experiments/baseline/evidence/main-run-715121-prompts.json").read_text(encoding="utf-8"))
    cache = LatentCache(ROOT / state["config"]["data"]["cache"], "dev")
    if selected["cache_fingerprint"] != cache.fingerprint or state["cache_fingerprint"] != cache.fingerprint:
        raise ValueError("Input representations differ")
    codec = Codec(device="cuda")
    if codec.fingerprint != state["codec"]:raise ValueError("Codec differs")
    model = Denoiser(state["config"]["model"]).cuda().eval(); model.load_state_dict(state["ema"])
    mean = torch.tensor(state["normalizer"]["mean"], dtype=torch.float32, device="cuda"); scale = state["normalizer"]["scale"]
    prior_pairs = [json.loads(line) for line in (ROOT / "docs/experiments/history-reencoding/evidence/history-reencoding-715121-pairs.jsonl").read_text(encoding="utf-8").splitlines()]
    controls = {(r["doc"], r["seed"]):r for r in prior_pairs if r["units"] == 128}
    prior_vectors = np.load(ROOT / "runs/reencode_history_715121/latents.npz")
    old_probe = json.loads((ROOT / "runs/review_715121/sampling_steps/samples.json").read_text(encoding="utf-8"))
    old = {(r["doc"], r["seed"], r["steps"]):r for r in old_probe}
    offsets = np.fromfile(cache.root / "unit_offsets.u64", dtype="<u8"); blob = (cache.root / "units.bin").read_bytes()
    counts = np.fromfile(cache.root / "train_counts.u64", dtype="<u8")
    seen = {blob[int(offsets[i]):int(offsets[i + 1])] for i in np.flatnonzero(counts)}
    protocol = {"date": "2026-11-01", "checkpoint_sha256":checkpoint_sha, "checkpoint_step":state["state"]["step"],
                "weights":"ema", "history":"continuous", "sample_precision":"fp16", "codec_precision":"fp32",
                "steps":[16,32,64], "prompts":16, "seeds":[42,123], "prefix_units":32, "generated_units":128,
                "cache_fingerprint":cache.fingerprint, "timing_prompts":4, "timing_seed":42,
                "source_sha256":{name:sha256(ROOT/name) for name in ["char_diff/flow.py","char_diff/codec.py","scripts/compare_sampling_steps.py"]}}
    write_json(out / "protocol.json", protocol)
    # Warm the model and kernels using a separate noise generator.
    with torch.autocast("cuda", dtype=torch.float16):
        generate(model, cache.story(selected["prompts"][0]["doc"])[:32], 32, 4, 999)
    torch.cuda.synchronize()
    records = []; vectors = {}; validations = []; started = time.perf_counter()
    timing_orders = [[16,32,64],[32,64,16],[64,16,32],[16,64,32]]
    with open(out / "samples.jsonl", "x", encoding="utf-8", buffering=1) as journal:
        for prompt_index, prompt in enumerate(selected["prompts"]):
            doc = prompt["doc"]; prefix = cache.story(doc)[:32]
            for seed in [42,123]:
                control = controls[(doc,seed)]
                if control["prompt"] != prompt["prompt"]:raise ValueError("Cached prefix differs")
                timing = prompt_index < 4 and seed == 42
                order = timing_orders[prompt_index] if timing else [16,32,64]
                for steps in order:
                    key = f"doc_{doc}_seed_{seed}_steps_{steps}"
                    fresh = steps != 32 or timing
                    if fresh:
                        torch.cuda.synchronize(); tick = time.perf_counter()
                        with torch.autocast("cuda", dtype=torch.float16):
                            z = generate(model, prefix, 128, steps, seed)
                        if not torch.isfinite(z).all():raise FloatingPointError("Non-finite output")
                        pieces = codec.decode(z*scale+mean); raw = b"".join(pieces)
                        torch.cuda.synchronize(); seconds = time.perf_counter()-tick
                        array = z.cpu().numpy()
                        if timing:
                            previous = control["arms"]["continuous"] if steps == 32 else old[(doc,seed,steps)]
                            if previous["generated_hex"] != raw.hex():raise AssertionError("Old sampler output was not reproduced")
                            if steps == 32 and not np.array_equal(array, prior_vectors[f"case_{control['case']}_continuous"]):
                                raise AssertionError("Cached continuous trajectory differs")
                            validations.append({"doc":doc,"seed":seed,"steps":steps,"byte_exact":True})
                    else:
                        previous = control["arms"]["continuous"]; raw = bytes.fromhex(previous["generated_hex"])
                        ends = np.r_[0,np.cumsum(previous["unit_lengths"])]
                        if ends[-1] != len(raw):raise ValueError("Cached byte offsets differ")
                        pieces = [raw[int(a):int(b)] for a,b in zip(ends[:-1],ends[1:])]
                        seconds = None; array = prior_vectors[f"case_{control['case']}_continuous"]
                    try:text = raw.decode("utf-8"); valid_utf8 = True
                    except UnicodeDecodeError:text = raw.decode("utf-8",errors="replace"); valid_utf8 = False
                    record = {"doc":doc,"seed":seed,"steps":steps,"prompt":prompt["prompt"],"generated":text,
                              "generated_hex":raw.hex(),"unit_lengths":[len(v) for v in pieces],"utf8_valid":valid_utf8,
                              "seen_training_units":sum(v in seen for v in pieces),"empty_units":sum(not v for v in pieces),
                              "units":128,"denoiser_calls":8*steps,"source":"fresh" if fresh else "validated_cached_control",
                              "seconds":seconds,"timing_case":timing}
                    records.append(record); vectors[key] = array
                    journal.write(json.dumps(record,ensure_ascii=False)+"\n")
                    print(json.dumps({k:record[k] for k in ["doc","seed","steps","source","seen_training_units","utf8_valid"]}),flush=True)
    groups = {}
    for steps in [16,32,64]:
        group = [r for r in records if r["steps"]==steps]; timing = [r["seconds"] for r in group if r["timing_case"]]
        groups[str(steps)] = {"cases":len(group),"units":128*len(group),"seen_training_units":sum(r["seen_training_units"] for r in group),
                              "seen_training_unit_rate":sum(r["seen_training_units"] for r in group)/(128*len(group)),
                              "valid_utf8":sum(r["utf8_valid"] for r in group),"empty_units":sum(r["empty_units"] for r in group),
                              "timing_cases":len(timing),"timing_seconds_mean":float(np.mean(timing)),"timing_seconds_median":float(np.median(timing)),
                              "denoiser_calls_per_sample":8*steps}
    mapping = {(r["doc"],r["seed"],r["steps"]):r for r in records}; paired = {}
    for steps in [16,64]:
        doc_deltas = []; counts_delta = []
        for prompt in selected["prompts"]:
            ds=[]
            for seed in [42,123]:
                delta = mapping[(prompt["doc"],seed,steps)]["seen_training_units"] - mapping[(prompt["doc"],seed,32)]["seen_training_units"]
                counts_delta.append(delta); ds.append(delta/128)
            doc_deltas.append(np.mean(ds))
        values=np.array(doc_deltas);idx=np.random.default_rng(20261101).integers(len(values),size=(10000,len(values)))
        paired[str(steps)]={"rate_delta_vs32":float(values.mean()),"story_bootstrap_95ci":np.quantile(values[idx].mean(1),[.025,.975]).tolist(),
                            "higher":sum(v>0 for v in counts_delta),"same":sum(v==0 for v in counts_delta),"lower":sum(v<0 for v in counts_delta)}
    np.savez(out / "latents.npz",**vectors)
    summary={"protocol":protocol,"groups":groups,"paired_vs32":paired,"validations":validations,
             "fresh_samples":sum(r["source"]=="fresh" for r in records),"cached_samples":sum(r["source"]!="fresh" for r in records),
             "elapsed_seconds":time.perf_counter()-started}
    write_json(out / "summary.json",summary)
    lines=["# 16/32/64 步：固定前缀逐组三档输出","","同一最终 EMA、连续历史、初始噪声和 128-unit 预算。",""]
    for i,prompt in enumerate(selected["prompts"],1):
        for seed in [42,123]:
            lines.extend([f"## {i}. doc={prompt['doc']} / seed={seed}","","**前缀**","",prompt["prompt"],""])
            for steps in [16,32,64]:lines.extend([f"**每块 {steps} 步**","",mapping[(prompt["doc"],seed,steps)]["generated"],""])
    (out / "samples.md").write_text("\n".join(lines),encoding="utf-8")
    print(json.dumps(summary),flush=True)


if __name__ == "__main__":main()
