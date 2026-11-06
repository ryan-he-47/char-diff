"""Summarize complete training steps after warmup, including failed-update time."""
import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from char_diff.utils import ROOT, sha256, write_json


def summarize(run, warmup=50):
    run = Path(run)
    if warmup < 0:
        raise ValueError("Warmup must be nonnegative")
    cfg = json.loads((run / "config.json").read_text(encoding="utf-8"))
    rows = [json.loads(line) for line in (run / "train.jsonl").read_text().splitlines() if line.strip()]
    if any(b["attempt"] <= a["attempt"] for a, b in zip(rows, rows[1:])):
        raise ValueError("Log contains overlapping restart segments; summarize a fresh calibration run")
    start = 0
    if warmup:
        start = next((i + 1 for i, row in enumerate(rows) if row["updated"] and row["step"] >= warmup), len(rows))
    steady = rows[start:]
    if not steady:
        raise ValueError("No measured updates remain after warmup")
    elapsed = sum(row["seconds"] for row in steady)
    units = sum(row["target_units"] for row in steady if row["updated"])
    rate = units / elapsed
    manifest_path = ROOT / cfg["data"]["cache"] / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    epoch_units = manifest["splits"]["train"]["units"]
    result = {
        "run": str(run.resolve()), "warmup_successful_updates": warmup,
        "measured_successful_updates": sum(bool(row["updated"]) for row in steady),
        "measured_skipped_updates": sum(not row["updated"] for row in steady),
        "measured_step_seconds": elapsed, "measured_target_units": units,
        "target_units_per_second": rate,
        "physical_positions_per_second": sum(row["physical_positions"] for row in steady) / elapsed,
        "step_seconds_median": statistics.median(row["seconds"] for row in steady),
        "peak_allocated_cuda_bytes": max(row["peak_cuda_bytes"] for row in rows),
        "gradient_norm_max": max(row["grad_norm"] for row in steady),
        "first20_mean_loss": statistics.mean(row["loss"] for row in rows[:20]),
        "last20_mean_loss": statistics.mean(row["loss"] for row in rows[-20:]),
        "epoch_units": epoch_units, "two_epoch_compute_hours": 2 * epoch_units / rate / 3600 if rate else None,
        "timing_scope": "batch construction, corruption, forward, backward, optimizer, EMA and CUDA sync; checkpoint/evaluation overhead is reported separately",
        "cache_fingerprint": sha256(manifest_path), "precision": cfg["train"]["precision"],
        "batch_windows": cfg["train"]["batch_windows"], "micro_batch": cfg["train"]["micro_batch"],
    }
    completion = run / "result.json"
    if completion.exists():
        result["completion"] = json.loads(completion.read_text())
    checkpoint = run / "last.pt"
    if checkpoint.exists():
        result["checkpoint_bytes"] = checkpoint.stat().st_size
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run", nargs="?", default="runs/v100_probe")
    p.add_argument("--warmup", type=int, default=50)
    p.add_argument("--output", default="runs/probe-summary.json")
    args = p.parse_args()
    report = summarize(ROOT / args.run, args.warmup)
    write_json(args.output, report)
    print(json.dumps(report, indent=2))
