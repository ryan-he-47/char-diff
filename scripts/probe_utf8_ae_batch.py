"""Measure the real FP16 AE batch capacity and throughput on an allocated GPU.

Each candidate runs in a fresh process, so an OOM cannot poison later probes.
Capacity uses 32-byte valid ASCII strings and the complete training loss. An
optional real-data pass checks the selected batch's sampling and end-to-end rate.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from char_diff.utils import ROOT, load_config, sha256, write_json


def worker(args):
    import torch
    import torch.nn.functional as F
    from char_diff.ae_quality import tensorize, targets_from_lengths
    from char_diff.ae_windows import UTF8WindowSampler
    from char_diff.vendor.bltz.config import Cfg
    from char_diff.vendor.bltz.models.autoencoder import ByteStringAE

    cfg = load_config(args.config)
    tc, dc = cfg["train"], cfg["data"]
    torch.set_num_threads(4)
    torch.manual_seed(tc["seed"])
    torch.cuda.manual_seed_all(tc["seed"])
    torch.cuda.init()
    device = torch.device("cuda")
    free_before, total = torch.cuda.mem_get_info()
    batch = args.worker_batch
    result = {"batch": batch, "kind": args.worker_kind,
              "device": torch.cuda.get_device_name(0), "torch_version": torch.__version__,
              "cuda_version": torch.version.cuda, "total_memory_bytes": total,
              "free_before_bytes": free_before, "precision": "fp16",
              "warmup_updates": args.warmup, "measured_updates": args.steps,
              "input_protocol": ("all positions valid, 32 ASCII bytes plus EOS" if args.worker_kind == "capacity"
                                 else "approved UTF-8 sampler, variable true lengths padded to 32 bytes"),
              "gradient_checkpointing": False}
    sampler = None
    try:
        model = ByteStringAE(Cfg({"model": cfg["model"]})).to(device).train()
        result["parameters"] = sum(p.numel() for p in model.parameters())
        optimizer = torch.optim.AdamW(model.parameters(), lr=tc["lr"], betas=tuple(tc["betas"]),
                                     eps=tc["eps"], weight_decay=tc["weight_decay"])
        scaler = torch.amp.GradScaler("cuda", init_scale=1024, growth_interval=500)
        if args.worker_kind == "real":
            init_start = time.perf_counter()
            sampler = UTF8WindowSampler(ROOT / args.data, seed=tc["seed"], max_bytes=cfg["model"]["l_max"],
                                        preload_texts=dc.get("preload_texts", False),
                                        **{k: dc[k] for k in ("augment_probability", "replace_probability",
                                                             "single_character_probability", "cache_documents")})
            result["sampler_initialization_seconds"] = time.perf_counter() - init_start
            result["preload_texts"] = dc.get("preload_texts", False)
            result["cache_documents"] = dc["cache_documents"]
            result["data_sha256"] = sampler.source_sha256
        else:
            # Full-width legitimate UTF-8 inputs, without CPU tensorization cost.
            ids = torch.randint(32, 127, (batch, cfg["model"]["l_max"]), device=device)
            lens = torch.full((batch,), cfg["model"]["l_max"], dtype=torch.long, device=device)
            mask = torch.zeros_like(ids, dtype=torch.bool)
            targets = targets_from_lengths(ids, lens)
        torch.cuda.reset_peak_memory_stats()
        timings, preparation, losses, norms, byte_counts = [], [], [], [], []
        for step in range(args.warmup + args.steps):
            torch.cuda.synchronize()
            start = time.perf_counter()
            if sampler is not None:
                units = [s.data for s in sampler.sample_batch(batch)]
                target_bytes = sum(map(len, units))
                ids, lens, mask = tensorize(units, cfg["model"]["l_max"], device)
                targets = targets_from_lengths(ids, lens)
            else:
                target_bytes = batch * cfg["model"]["l_max"]
            torch.cuda.synchronize()
            ready = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits = model(ids, mask)
            loss = F.cross_entropy(logits.float().flatten(0, 1), targets.flatten(),
                                   ignore_index=-100, reduction="sum") / (target_bytes + batch)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), tc["clip"]))
            loss_value = float(loss.detach())
            finite = math.isfinite(loss_value) and math.isfinite(norm) and norm <= tc["max_grad_norm"]
            if not finite:
                raise FloatingPointError(f"Non-finite/oversized gradient: loss={loss_value}, norm={norm}")
            scaler.step(optimizer)
            scaler.update()
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            if step >= args.warmup:
                timings.append(elapsed)
                preparation.append(ready - start)
                losses.append(loss_value)
                norms.append(norm)
                byte_counts.append(target_bytes)
        peak_allocated = torch.cuda.max_memory_allocated()
        peak_reserved = torch.cuda.max_memory_reserved()
        result.update(status="ok", all_gradients_finite=True,
                      peak_allocated_bytes=peak_allocated, peak_reserved_bytes=peak_reserved,
                      allocated_fraction_available=peak_allocated / free_before,
                      reserved_fraction_available=peak_reserved / free_before,
                      allocated_fraction_total=peak_allocated / total,
                      reserved_fraction_total=peak_reserved / total,
                      step_seconds=timings, median_step_seconds=statistics.median(timings),
                      preparation_seconds=preparation, losses=losses, grad_norms=norms,
                      target_bytes=byte_counts,
                      samples_per_second=(batch * len(timings)) / sum(timings),
                      target_bytes_per_second=sum(byte_counts) / sum(timings))
    except (torch.cuda.OutOfMemoryError, RuntimeError, FloatingPointError) as exc:
        is_oom = isinstance(exc, torch.cuda.OutOfMemoryError) or "out of memory" in str(exc).lower()
        result.update(status="oom" if is_oom else "error", error=f"{type(exc).__name__}: {exc}"[:1000])
        if is_oom:
            result.update(peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                          peak_reserved_bytes=torch.cuda.max_memory_reserved())
    finally:
        if sampler is not None:
            sampler.close()
    write_json(args.output, result)
    print(json.dumps({k: result[k] for k in ("batch", "kind", "status")}))
    return 0 if result["status"] in ("ok", "oom") else 1


def choose(results, target_fraction, max_fraction, throughput_tolerance):
    safe = [r for r in results if r["status"] == "ok" and
            r["reserved_fraction_available"] <= max_fraction]
    filled = [r for r in safe if r["allocated_fraction_available"] >= target_fraction]
    if not filled:
        return None
    best_rate = max(r["samples_per_second"] for r in filled)
    useful = [r for r in filled if r["samples_per_second"] >= best_rate * (1 - throughput_tolerance)]
    return max(useful, key=lambda r: r["batch"])


def probe(args):
    cfg = load_config(args.config)
    if cfg["model"]["l_max"] != 32 or cfg["model"]["d_emb"] != 48:
        raise ValueError("This deployment probe expects the approved 32-byte / 48-dimensional AE")
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(f"Probe output already exists: {output}")
    records_dir = output.parent / (output.stem + "_candidates")
    records_dir.mkdir(parents=True, exist_ok=False)
    results = []
    report = {"format_version": 1, "status": "probing", "config": str(Path(args.config).resolve()),
              "config_sha256": sha256(args.config), "results": results,
              "selection": {"target_allocated_fraction_available": args.target_fraction,
                            "max_reserved_fraction_available": args.max_fraction,
                            "throughput_tolerance": args.throughput_tolerance,
                            "rule": "largest filled safe candidate within tolerance of best filled throughput"}}

    def run(batch, kind="capacity"):
        path = records_dir / f"{kind}_{batch}.json"
        command = [sys.executable, str(Path(__file__).resolve()), "--worker-batch", str(batch),
                   "--worker-kind", kind, "--config", str(Path(args.config).resolve()),
                   "--output", str(path), "--warmup", str(args.warmup), "--steps", str(args.steps)]
        if args.data:
            command += ["--data", args.data]
        proc = subprocess.run(command, text=True, capture_output=True)
        if not path.is_file():
            row = {"batch": batch, "kind": kind, "status": "error",
                   "returncode": proc.returncode, "error": proc.stderr[-2000:] or proc.stdout[-2000:]}
        else:
            row = json.loads(path.read_text(encoding="utf-8"))
            row["returncode"] = proc.returncode
        results.append(row)
        write_json(output, report)
        print(json.dumps({k: row[k] for k in ("batch", "kind", "status")}, ensure_ascii=False), flush=True)
        if row["status"] == "error":
            raise RuntimeError(f"Candidate {batch} failed: {row['error']}")
        return row

    try:
        candidates = sorted(set(b for b in args.candidates if b <= args.max_batch))
        if not candidates:
            raise ValueError("No candidates within max_batch")
        lower, upper = 0, None
        for batch in candidates:
            row = run(batch)
            if row["status"] == "oom" or row.get("reserved_fraction_available", 1.) > args.max_fraction:
                upper = batch
                break
            lower = batch
        while upper is None and lower < args.max_batch:
            batch = min(args.max_batch, lower * 2)
            row = run(batch)
            if row["status"] == "oom" or row.get("reserved_fraction_available", 1.) > args.max_fraction:
                upper = batch
                break
            lower = batch
        # At most a few localized probes after the first unsafe point; capacities
        # are measured, not inferred from parameter count or a smaller device.
        if upper is not None:
            for _ in range(args.refine):
                middle = ((lower + upper) // (2 * args.granularity)) * args.granularity
                if middle <= lower or middle >= upper or middle < 1:
                    break
                row = run(middle)
                if row["status"] == "oom" or row.get("reserved_fraction_available", 1.) > args.max_fraction:
                    upper = middle
                else:
                    lower = middle
        selected = choose([r for r in results if r["kind"] == "capacity"], args.target_fraction,
                          args.max_fraction, args.throughput_tolerance)
        if selected is None:
            report["status"] = "needs_capacity_review"
            report["reason"] = "No measured candidate meets both occupancy and memory headroom requirements"
            write_json(output, report)
            return 2
        if args.data:
            real = run(selected["batch"], "real")
            if real["status"] != "ok" or real["reserved_fraction_available"] > args.max_fraction:
                report.update(status="needs_capacity_review", reason="Selected batch failed real-data headroom check")
                write_json(output, report)
                return 2
            report["selected_real_data"] = real
        micro = selected["batch"]
        global_batch = max(int(cfg["train"]["batch_windows"]), micro)
        report.update(status="complete", selected_micro_batch=micro,
                      selected_batch_windows=global_batch, selected_capacity=selected,
                      effective_batch_changed=(global_batch != cfg["train"]["batch_windows"]))
        if args.deploy_config:
            import yaml
            deploy = copy.deepcopy(cfg)
            deploy["train"].update(batch_windows=global_batch, micro_batch=micro)
            if args.data:
                deploy["data"]["train"] = args.data
            if args.run_dir:
                deploy["train"]["run_dir"] = args.run_dir
            if args.require_quota:
                deploy["storage"]["require_quota"] = True
            path = Path(args.deploy_config)
            if path.exists():
                raise FileExistsError(f"Deployment config already exists: {path}")
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("x", encoding="utf-8") as f:
                yaml.safe_dump(deploy, f, sort_keys=False, allow_unicode=True)
            report["deploy_config"] = str(path.resolve())
            report["deploy_config_sha256"] = sha256(path)
        write_json(output, report)
        print(json.dumps({k: report[k] for k in ("status", "selected_micro_batch", "selected_batch_windows")}), flush=True)
        return 0
    except BaseException as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        write_json(output, report)
        raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/tinystories_multilingual_ae48.yaml")
    p.add_argument("--output", required=True)
    p.add_argument("--data", help="Prepared training JSONL, for a real-data verification of the selected batch")
    p.add_argument("--deploy-config")
    p.add_argument("--run-dir")
    p.add_argument("--require-quota", action="store_true")
    p.add_argument("--candidates", type=int, nargs="+", default=[512, 1024, 2048, 4096, 8192, 12288, 16384, 24576, 32768, 49152, 65536])
    p.add_argument("--max-batch", type=int, default=65536)
    p.add_argument("--target-fraction", type=float, default=.80)
    p.add_argument("--max-fraction", type=float, default=.92)
    p.add_argument("--throughput-tolerance", type=float, default=.05)
    p.add_argument("--granularity", type=int, default=512)
    p.add_argument("--refine", type=int, default=3)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--steps", type=int, default=2)
    p.add_argument("--worker-batch", type=int, help=argparse.SUPPRESS)
    p.add_argument("--worker-kind", choices=["capacity", "real"], default="capacity", help=argparse.SUPPRESS)
    args = p.parse_args()
    if args.warmup < 1 or args.steps < 2 or args.granularity < 1 or args.refine < 0:
        p.error("Require >=1 warmup, >=2 measured updates, positive granularity and nonnegative refinement")
    if not 0 <= args.target_fraction < args.max_fraction < 1 or not 0 <= args.throughput_tolerance < 1:
        p.error("Invalid utilization/headroom/throughput fractions")
    if args.max_batch < 1 or any(b < 1 for b in args.candidates):
        p.error("Batch sizes must be positive")
    if args.worker_batch:
        if args.worker_kind == "real" and not args.data:
            p.error("Real-data worker requires --data")
        return worker(args)
    return probe(args)


if __name__ == "__main__":
    raise SystemExit(main())
