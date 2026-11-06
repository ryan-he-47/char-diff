"""UTF-8 window AE training, with byte exposure accounting and complete resume."""
from __future__ import annotations

import copy
import hashlib
import json
import math
import random
import signal
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .ae_quality import tensorize, targets_from_lengths
from .ae_windows import UTF8WindowSampler
from .storage import StorageGuard
from .training import atomic_checkpoint, rng_state, restore_rng
from .utils import ROOT, sha256, write_json
from .vendor.bltz.config import Cfg
from .vendor.bltz.models.autoencoder import ByteStringAE


def learning_rate(state, cfg, update_bytes=0):
    """WSD with an open-ended stable phase and an explicitly forked linear decay.

    Decay uses successful bytes. The prospective batch end reaches the exact
    floor on the last update; skipped updates leave the decay cursor unchanged.
    Job limits never enter this function.
    """
    if state["phase"] == "cooldown":
        progress = min(1., max(0., (state["seen_bytes"] + update_bytes - state["decay_start_bytes"]) /
                              state["decay_budget_bytes"]))
        if progress >= 1.:
            return cfg["final_lr"]
        if progress <= 0.:
            return cfg["lr"]
        return cfg["lr"] + (cfg["final_lr"] - cfg["lr"]) * progress
    warm = cfg["warmup_updates"]
    if state["step"] < warm:
        return cfg["lr"] * (state["step"] + 1) / warm
    return cfg["lr"]


def _recipe(cfg, sampler):
    operational = {"run_dir", "log_every", "save_every", "run_bytes", "cooldown_bytes",
                   "diagnostic_bytes", "stage_every_bytes", "keep_stages"}
    tc = {k: v for k, v in cfg["train"].items() if k not in operational}
    return {"model": cfg["model"], "train": tc, "sampling": sampler.protocol,
            "objective": "sum_byte_and_eos_ce / valid_byte_and_eos_count", "version": 2}


def train(cfg, resume=None, stop_after=0, device=None, stop_at_bytes=0, cooldown_from=None):
    tc, dc = cfg["train"], cfg["data"]
    if resume and cooldown_from:
        raise ValueError("Choose exact resume or a new cooldown branch")
    if tc.get("schedule") != "wsd":
        raise ValueError("This AE trainer uses WSD; legacy cosine smoke checkpoints remain separate")
    for key in ("batch_windows", "micro_batch", "run_bytes", "cooldown_bytes", "stage_every_bytes",
                "keep_stages", "log_every", "save_every", "max_consecutive_skips"):
        if int(tc[key]) < 1:
            raise ValueError(f"train.{key} must be positive")
    if min(stop_after, stop_at_bytes, tc["warmup_updates"], tc["diagnostic_bytes"]) < 0:
        raise ValueError("Stop limits and warmup must be nonnegative")
    if not 0 <= tc["final_lr"] <= tc["lr"] or not tc["lr"] > 0:
        raise ValueError("Require 0 <= final_lr <= positive lr")
    if tc["precision"] not in ("fp16", "fp32"):
        raise ValueError("AE training supports fp16 or fp32")
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    torch.set_num_threads(4)
    random.seed(tc["seed"])
    np.random.seed(tc["seed"])
    torch.manual_seed(tc["seed"])
    if device.type == "cuda":
        torch.cuda.manual_seed_all(tc["seed"])
    run = ROOT / tc["run_dir"]
    if not resume and run.exists() and any(run.iterdir()):
        raise FileExistsError(f"Fresh run requires an empty directory: {run}")
    if resume and run.exists() and any(run.iterdir()) and (ROOT / resume).resolve().parent != run.resolve():
        raise ValueError("Resume into an empty run or the source checkpoint's run directory")
    source = ROOT / dc["train"]
    if dc.get("manifest") and not dc.get("full_manifest"):
        manifest_path = ROOT / dc["manifest"]
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        artifact = manifest.get("artifacts", {}).get("train", {})
        if manifest.get("status") != "complete" or artifact.get("sha256") != sha256(source):
            raise ValueError("Training pool requires a complete, hash-matching manifest")
    guard = StorageGuard(ROOT, limit_gib=cfg["storage"]["project_limit_gib"],
                         reserve_gib=cfg["storage"]["reserve_gib"],
                         require_quota=cfg["storage"].get("require_quota", False))
    one_pass = bool(dc.get("full_manifest") and tc.get("one_pass"))
    if one_pass and cooldown_from:
        raise ValueError("The full pass owns its final 10% decay")
    model = ByteStringAE(Cfg({"model": cfg["model"]})).to(device)
    params = sum(p.numel() for p in model.parameters())
    estimate = params * 16 + 2 * 1024**2
    guard.check(3 * estimate)
    sample_options = {key: dc[key] for key in ("augment_probability", "replace_probability", "single_character_probability")}
    if dc.get("full_manifest"):
        from .ae_full_sampler import FullCorpusSampler
        options = dict(seed=tc["seed"],max_bytes=cfg["model"]["l_max"],**sample_options)
        if one_pass:
            from .ae_prefetch import PrefetchedEpoch, epoch_lr
            sampler = PrefetchedEpoch(ROOT / dc["full_manifest"], options)
        else:
            sampler = FullCorpusSampler(ROOT / dc["full_manifest"], **options)
    else:
        sampler = UTF8WindowSampler(source, seed=tc["seed"], max_bytes=cfg["model"]["l_max"],
                                    preload_texts=dc.get("preload_texts", False),
                                    cache_documents=dc["cache_documents"], **sample_options)
    recipe = _recipe(cfg, sampler)
    optimizer = torch.optim.AdamW(model.parameters(), lr=tc["lr"], betas=tuple(tc["betas"]),
                                 eps=tc["eps"], weight_decay=tc["weight_decay"])
    amp = device.type == "cuda" and tc["precision"] == "fp16"
    scaler = torch.amp.GradScaler("cuda", enabled=amp, init_scale=1024, growth_interval=500)
    state = {"step": 0, "attempts": 0, "seen_bytes": 0, "attempted_bytes": 0,
             "skipped": 0, "consecutive_skips": 0, "warmup_end_bytes": 0,
             "diagnostic_saved": False, "phase": "stable", "decay_start_bytes": None,
             "decay_budget_bytes": None, "last_stage_boundary": 0, "stage_checkpoints": []}
    origin = {"kind": "random", "checkpoint": None, "sha256": None}
    if dc.get("full_manifest") and not (resume or cooldown_from):
        parent = ROOT / dc["init_stable"]
        if sha256(parent) != dc["init_sha256"]:
            raise ValueError("Full corpus continuation source checksum mismatch")
        saved = torch.load(parent, map_location="cpu", weights_only=False)
        if saved["state"]["phase"] != "stable" or saved["cfg"]["model"] != cfg["model"]:
            raise ValueError("Expected compatible stable AE checkpoint")
        model.load_state_dict(saved["model"],strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        scaler.load_state_dict(saved["scaler"])
        origin = {"kind":"full_corpus_continuation","checkpoint":str(parent),
                  "sha256":dc["init_sha256"],"source_step":saved["step"],"source_bytes":saved["state"]["seen_bytes"]}
        del saved
    if resume or cooldown_from:
        source_checkpoint = ROOT / (resume or cooldown_from)
        # Hash and load the same open file, including when a stable job rotates
        # last.pt concurrently with a cooldown launch.
        with open(source_checkpoint, "rb") as handle:
            digest = hashlib.sha256()
            for chunk in iter(lambda: handle.read(8 * 1024**2), b""):
                digest.update(chunk)
            handle.seek(0)
            saved = torch.load(handle, map_location="cpu", weights_only=False)
        checkpoint_sha256 = digest.hexdigest()
        if saved.get("recipe") != recipe:
            raise ValueError("WSD resume/branch requires matching model, sampling and training recipe (version 2)")
        model.load_state_dict(saved["model"], strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        scaler.load_state_dict(saved["scaler"])
        sampler.load_state_dict(saved["sampler"])
        state = copy.deepcopy(saved["state"])
        origin = copy.deepcopy(saved["origin"])
        if cooldown_from:
            if state["phase"] != "stable" or state["step"] < tc["warmup_updates"]:
                raise ValueError("Cooldown must branch from a stable checkpoint after warmup")
            state.update(phase="cooldown", decay_start_bytes=state["seen_bytes"],
                         decay_budget_bytes=tc["cooldown_bytes"], stage_checkpoints=[],
                         last_stage_boundary=(state["seen_bytes"] // tc["stage_every_bytes"]) * tc["stage_every_bytes"])
            origin = {"kind": "cooldown", "checkpoint": str(source_checkpoint.resolve()),
                      "sha256": checkpoint_sha256, "source_step": state["step"],
                      "source_bytes": state["seen_bytes"]}
        elif not one_pass and state["phase"] == "cooldown" and tc["cooldown_bytes"] != state["decay_budget_bytes"]:
            raise ValueError("Cooldown resume preserves its original cooldown_bytes")
        restore_rng(saved["rng"])
    start_step = state["step"]
    invocation_end = state["seen_bytes"] + tc["run_bytes"]
    run.mkdir(parents=True, exist_ok=True)
    write_json(run / "config.json", cfg)
    write_json(run / "recipe.json", {"recipe": recipe, "parameters": params, "origin": origin})
    stop = {"requested": False}
    previous_handlers = {}
    for signum in (signal.SIGINT, signal.SIGTERM):
        previous_handlers[signum] = signal.getsignal(signum)
        signal.signal(signum, lambda *_: stop.update(requested=True))
    last_saved_attempt = -1

    def save(name="last.pt", rotate=True):
        nonlocal last_saved_attempt
        payload = {"cfg": {"model": cfg["model"]}, "model": model.state_dict(), "step": state["step"],
                   "config": cfg, "recipe": recipe, "state": state, "origin": origin, "optimizer": optimizer.state_dict(),
                   "scaler": scaler.state_dict(), "sampler": sampler.state_dict(), "rng": rng_state()}
        atomic_checkpoint(run / name, payload, guard, estimate, rotate=rotate)
        if name == "last.pt":
            last_saved_attempt = state["attempts"]

    def save_stage():
        boundary = (state["seen_bytes"] // tc["stage_every_bytes"]) * tc["stage_every_bytes"]
        if boundary <= state["last_stage_boundary"]:
            return
        name = f"stage_{boundary:012d}.pt"
        old = state["stage_checkpoints"]
        retained = (old + [name])[-tc["keep_stages"]:]
        state.update(last_stage_boundary=boundary, stage_checkpoints=retained)
        save(name, rotate=False)
        # Only prune this run's checkpoint names recorded by this writer, after
        # the replacement is durable. Never recurse or touch other artifacts.
        for obsolete in old:
            path = run / obsolete
            if obsolete not in retained and Path(obsolete).name == obsolete and obsolete.startswith("stage_") \
                    and obsolete.endswith(".pt") and path.resolve().parent == run.resolve() and path.is_file():
                path.unlink()

    def emit(log, row):
        if device.type == "cuda":
            row.update(cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
                       cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved(device),
                       cuda_total_memory_bytes=torch.cuda.get_device_properties(device).total_memory)
        log.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        log.flush()
        print(json.dumps(row, ensure_ascii=False, allow_nan=False), flush=True)

    monitor = None
    if one_pass and device.type == "cuda" and tc.get("gpu_monitor", True):
        from .gpu_monitor import GPUMonitor
        monitor = GPUMonitor(run / "gpu_samples.jsonl", interval=1).start()
    held_batch = None
    wait_seconds = 0.
    model.train()
    started = time.monotonic()
    try:
        with open(run / "train.jsonl", "a", encoding="utf-8") as log:
            emit(log, {"event": "start", "resume": str(resume) if resume else None, "origin": origin,
                       "device": str(device), "parameters": params, "invocation_end_bytes": invocation_end, **state})
            pause_reason = None
            while True:
                if one_pass and sampler.stats.get("completed_epochs", 0):
                    if sampler.stats["natural_source_bytes"] != sampler.total_bytes:
                        raise RuntimeError("Source coverage does not match full manifest")
                    break
                if not one_pass and state["phase"] == "cooldown" and state["seen_bytes"] >= state["decay_start_bytes"] + state["decay_budget_bytes"]:
                    break
                if stop["requested"] or (stop_after and state["step"] - start_step >= stop_after):
                    pause_reason = "signal" if stop["requested"] else "stop_after"
                    break
                if stop_at_bytes and state["seen_bytes"] >= stop_at_bytes:
                    pause_reason = "stop_at_bytes"
                    break
                if not one_pass and state["seen_bytes"] >= invocation_end:
                    pause_reason = "run_bytes"
                    break
                if one_pass:
                    if held_batch is None:
                        tick = time.perf_counter()
                        held_batch = sampler.next_batch(tc["batch_windows"])
                        wait_seconds += time.perf_counter() - tick
                    packed, after = held_batch
                    batch_count = len(packed)
                    if not batch_count:
                        raise RuntimeError("Unexpected end before manifest byte count")
                    source_end = after["stats"]["natural_source_bytes"]
                    if state["phase"] == "stable" and 10 * source_end > 9 * sampler.total_bytes:
                        save("stable_end.pt", rotate=False)
                        state["phase"] = "cooldown"
                    lr = epoch_lr(source_end,sampler.total_bytes,tc["lr"],tc["final_lr"])
                    nbytes = int(packed[:,0].sum())
                    host = torch.from_numpy(packed.copy() if not packed.flags.writeable else packed)
                    if device.type == "cuda":
                        host = host.pin_memory()
                    inputs = host.to(device, non_blocking=True)
                else:
                    samples = sampler.sample_batch(tc["batch_windows"])
                    units = [s.data for s in samples]
                    batch_count = len(units)
                    nbytes = sum(map(len, units))
                    lr = learning_rate(state, tc, update_bytes=nbytes)
                denominator = nbytes + batch_count
                for group in optimizer.param_groups:
                    group["lr"] = lr
                optimizer.zero_grad(set_to_none=True)
                total_loss = 0.
                for first in range(0, batch_count, tc["micro_batch"]):
                    if one_pass:
                        chunk = inputs[first:first + tc["micro_batch"]]
                        lens = chunk[:,0].long()
                        ids = chunk[:,1:].long()
                        mask = torch.arange(ids.shape[1],device=device)[None] >= lens[:,None]
                    else:
                        ids, lens, mask = tensorize(units[first:first + tc["micro_batch"]], cfg["model"]["l_max"], device)
                    targets = targets_from_lengths(ids, lens)
                    with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                        logits = model(ids, mask)
                    loss = F.cross_entropy(logits.float().flatten(0, 1), targets.flatten(),
                                           ignore_index=-100, reduction="sum") / denominator
                    scaler.scale(loss).backward()
                    total_loss += float(loss.detach())
                scaler.unscale_(optimizer)
                norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), tc["clip"]))
                good = math.isfinite(total_loss) and math.isfinite(norm) and norm <= tc["max_grad_norm"]
                if good:
                    scaler.step(optimizer)
                scaler.update()
                state["attempts"] += 1
                state["attempted_bytes"] += nbytes
                if good:
                    if one_pass:
                        sampler.committed = after
                        state["natural_source_bytes"] = source_end
                        state["total_raw_bytes"] = sampler.total_bytes
                        held_batch = None
                    state["step"] += 1
                    state["seen_bytes"] += nbytes
                    state["consecutive_skips"] = 0
                    if state["step"] == tc["warmup_updates"]:
                        state["warmup_end_bytes"] = state["seen_bytes"]
                else:
                    state["skipped"] += 1
                    state["consecutive_skips"] += 1
                if state["attempts"] == 1 or state["attempts"] % tc["log_every"] == 0 or not good:
                    emit(log, {"event": "update", **state, "loss": total_loss if math.isfinite(total_loss) else None,
                               "grad_norm": norm if math.isfinite(norm) else None, "lr": lr,
                               "elapsed_seconds": time.monotonic() - started, "scale": scaler.get_scale(),
                               "sampling": sampler.stats, "data_wait_seconds":wait_seconds,
                               "gpu":monitor.summary() if monitor else {}})
                if tc["diagnostic_bytes"] and not state["diagnostic_saved"] and state["seen_bytes"] >= tc["diagnostic_bytes"]:
                    state["diagnostic_saved"] = True
                    save("diagnostic.pt", rotate=False)
                if good:
                    save_stage()
                if good and state["step"] % tc["save_every"] == 0:
                    save()
                if state["consecutive_skips"] >= tc["max_consecutive_skips"]:
                    save()
                    raise FloatingPointError("Consecutive invalid updates reached the protection limit; full state saved")
            if state["attempts"] != last_saved_attempt:
                save()
            result = {"event": "complete" if pause_reason is None else "paused", "pause_reason": pause_reason,
                      **state, "parameters": params, "elapsed_seconds": time.monotonic() - started,
                      "sampling": sampler.stats, "checkpoint": str(run / "last.pt")}
            emit(log, result)
            write_json(run / "status.json", result)
            return result
    finally:
        if monitor is not None:
            monitor.close()
        sampler.close()
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)
