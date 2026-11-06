"""Focused regression for the observed FP16 residual overflow and stop guard."""
import contextlib
import io
import json
import math
import sys
import tempfile
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from char_diff.data import LatentCache, WindowSampler
from char_diff.flow import corrupt
from char_diff.gmm import loss_sum
from char_diff.model import Denoiser
from char_diff.training import restore_rng, train
from char_diff.utils import ROOT, load_config, write_json


def check_stopped_models(folder):
    report = {}
    cache = LatentCache(ROOT / "data/tinystories")
    for arm in ("early", "late"):
        stopped = torch.load(folder / arm / "last.pt", map_location="cpu", weights_only=False)
        cfg = stopped["config"]
        model = Denoiser(cfg["model"]).cuda().eval()
        model.load_state_dict(stopped["model"])
        sampler = WindowSampler(len(cache), cfg["train"]["seed"])
        sampler.load_state_dict(stopped["sampler"])
        indices = sampler.next(cfg["train"]["batch_windows"])
        generator = torch.Generator(device="cuda").manual_seed(20261102)
        comparison = None
        for offset in range(0, len(indices), cfg["train"]["micro_batch"]):
            batch = cache.batch(indices[offset:offset + cfg["train"]["micro_batch"]], "cuda")
            x, t = corrupt(batch, generator=generator)
            args = (x, t, batch["roles"], batch["positions"], batch["lengths"], batch["allowed"])
            for block in model.blocks:
                block.fp32_residual = False
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
                old, count = loss_sum(model(*args), batch, cfg["model"]["gmm_sigma"])
            if torch.isfinite(old):
                continue
            for block in model.blocks:
                block.fp32_residual = True
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
                fixed, _ = loss_sum(model(*args), batch, cfg["model"]["gmm_sigma"])
            with torch.inference_mode():
                reference, _ = loss_sum(model(*args), batch, cfg["model"]["gmm_sigma"])
            assert torch.isfinite(fixed)
            torch.testing.assert_close(fixed, reference, rtol=.002, atol=.001)
            comparison = {"microbatch_offset": offset, "original_fp16_loss": "nan",
                          "fixed_amp_loss": float(fixed / (48 * count)),
                          "fp32_loss": float(reference / (48 * count))}
            break
        if comparison is None:
            # The early state sits close to the FP16 limit; its next batch may
            # stay finite on a different GPU. The late state is the reproducer.
            assert arm == "early", "The late overflow regression did not reproduce"
            for block in model.blocks:
                block.fp32_residual = True
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
                fixed, _ = loss_sum(model(*args), batch, cfg["model"]["gmm_sigma"])
            with torch.inference_mode():
                reference, _ = loss_sum(model(*args), batch, cfg["model"]["gmm_sigma"])
            torch.testing.assert_close(fixed, reference, rtol=.002, atol=.001)
            comparison = {"microbatch_offset": offset, "original_fp16_loss": float(old / (48 * count)),
                          "fixed_amp_loss": float(fixed / (48 * count)),
                          "fp32_loss": float(reference / (48 * count)), "overflow_reproduced": False}
        del stopped

        # Restore the preceding complete state, including its healthy GradScaler.
        checkpoint = torch.load(folder / arm / "previous.pt", map_location="cpu", weights_only=False)
        tc = checkpoint["config"]["train"]
        model.load_state_dict(checkpoint["model"])
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=tc["lr"], betas=tuple(tc["betas"]),
                                      eps=tc["eps"], weight_decay=tc["weight_decay"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scaler = torch.amp.GradScaler("cuda")
        scaler.load_state_dict(checkpoint["scaler"])
        sampler.load_state_dict(checkpoint["sampler"])
        restore_rng(checkpoint["rng"])
        rates = json.loads((ROOT / tc["reference_schedule"]).read_text())["learning_rates"]
        start_step = checkpoint["state"]["step"]
        updates = []
        for step in range(start_step, start_step + 2):
            indices = sampler.next(tc["batch_windows"])
            count = cache.target_units(indices)
            optimizer.zero_grad(set_to_none=True)
            for group in optimizer.param_groups:
                group["lr"] = rates[step]
            total = 0.0
            for offset in range(0, len(indices), tc["micro_batch"]):
                batch = cache.batch(indices[offset:offset + tc["micro_batch"]], "cuda")
                x, t = corrupt(batch)
                with torch.autocast("cuda", dtype=torch.float16):
                    raw, _ = loss_sum(model(x, t, batch["roles"], batch["positions"], batch["lengths"], batch["allowed"]),
                                      batch, cfg["model"]["gmm_sigma"])
                loss = raw / (48 * count)
                scaler.scale(loss).backward()
                total += float(loss.detach())
            scaler.unscale_(optimizer)
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), tc["clip"])
            assert torch.isfinite(norm) and float(norm) < 2000 and math.isfinite(total)
            scaler.step(optimizer)
            scaler.update()
            assert all(torch.isfinite(p).all() for p in model.parameters())
            updates.append({"step": step + 1, "loss": total, "grad_norm": float(norm), "lr": rates[step]})
        report[arm] = {"overflow_reproduction": comparison, "resume_source_step": start_step,
                       "updates": updates, "skipped": 0, "final_scale": scaler.get_scale()}
        print(json.dumps({"arm": arm, **report[arm]}), flush=True)
        del checkpoint, model, optimizer, scaler
        torch.cuda.empty_cache()
    return report


def check_invalid_stop():
    cfg = load_config(ROOT / "configs/tinystories_x0_22m.yaml")
    cfg["data"]["cache"] = "data/tiny_smoke"
    cfg["model"].update(width=32, layers=1, heads=2, ffn=64, grad_checkpoint=False)
    cfg["train"].update(batch_windows=2, micro_batch=2, precision="fp32", max_steps=1,
                        milestones=[], save_every=250, log_every=99)
    cfg["storage"]["require_quota"] = False
    original = Denoiser.forward
    def invalid(self, *args, **kwargs):
        return original(self, *args, **kwargs) * float("nan")
    with tempfile.TemporaryDirectory(prefix="gmm_guard_", dir=ROOT / "runs") as temp:
        cfg["train"]["run_dir"] = temp
        Denoiser.forward = invalid
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                try:
                    train(cfg, device="cpu")
                except RuntimeError as error:
                    assert "20 consecutive invalid" in str(error)
                else:
                    raise AssertionError("Expected the repeated-invalid guard")
        finally:
            Denoiser.forward = original
        saved = torch.load(Path(temp) / "last.pt", map_location="cpu", weights_only=False)
        rows = [json.loads(line) for line in (Path(temp) / "train.jsonl").read_text().splitlines()]
        assert len(rows) == 20 and all(not row["updated"] for row in rows)
        assert saved["state"]["step"] == 0 and saved["state"]["attempts"] == 20
        assert all(torch.isfinite(value).all() for key in ("model", "ema") for value in saved[key].values())
        assert not saved["optimizer"]["state"]
    print("PASS: 20 invalid attempts logged, zero updates, complete healthy recovery state saved", flush=True)
    return {"invalid_attempts_logged": 20, "updates": 0, "saved_model_and_ema_finite": True}


if __name__ == "__main__":
    torch.set_num_threads(4)
    report = {"device": torch.cuda.get_device_name(), "torch": torch.__version__}
    report["arms"] = check_stopped_models(ROOT / "runs/gmm_nan_review_20261102")
    report["invalid_guard"] = check_invalid_stop()
    write_json(ROOT / "docs/experiments/gmm-switch-timing/evidence/nan-recovery-tests.json", report)
