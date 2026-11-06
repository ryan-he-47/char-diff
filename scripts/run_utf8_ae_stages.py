"""Run an open-ended AE stable line with periodic validation and cooldowns."""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
import yaml
from char_diff.utils import ROOT, load_config, sha256, write_json

MIB = 1024**2
SCRIPTS = Path(__file__).resolve().parent


def now():
    return datetime.now(timezone.utc).isoformat()


def progress(checkpoint):
    if not checkpoint.is_file():
        return None
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = payload["state"]
    return {key: state.get(key) for key in (
        "step", "seen_bytes", "phase", "skipped", "decay_start_bytes", "decay_budget_bytes")}


def cooldown_due(target, interval):
    multiple, remainder = divmod(target, interval)
    return remainder == 0 and multiple > 0 and multiple & (multiple - 1) == 0


class ChildRunner:
    """TERM reaches the trainer's graceful-save handler, including wall-time stops."""
    def __init__(self, max_wall_seconds):
        self.deadline = time.monotonic() + max_wall_seconds
        self.stop_reason = None
        self.child = None
        self.handlers = {}

    def request_stop(self, reason):
        self.stop_reason = self.stop_reason or reason
        if self.child is not None and self.child.poll() is None:
            try:
                self.child.send_signal(signal.SIGTERM)
            except ProcessLookupError:
                pass

    def stopped(self):
        if time.monotonic() >= self.deadline and self.stop_reason is None:
            self.request_stop("wall_time")
        return self.stop_reason is not None

    def __enter__(self):
        for signum in (signal.SIGTERM, signal.SIGINT):
            self.handlers[signum] = signal.getsignal(signum)
            signal.signal(signum, lambda number, _: self.request_stop(f"signal_{number}"))
        return self

    def __exit__(self, *_):
        for signum, handler in self.handlers.items():
            signal.signal(signum, handler)

    def call(self, command):
        if self.stopped():
            return False
        print(json.dumps({"event": "launch", "at": now(), "command": command}), flush=True)
        self.child = subprocess.Popen(command)
        try:
            while self.child.poll() is None:
                self.stopped()
                time.sleep(.2)
            code = self.child.returncode
        finally:
            self.child = None
        if self.stopped():
            return False
        if code:
            raise subprocess.CalledProcessError(code, command)
        return True


def run_workflow(config_path, *, calibration, device="cuda", max_wall_seconds=27000,
                 samples=4096, eval_batch_size=4096, review_bytes=512 * MIB,
                 diagnostic_bytes=64 * MIB, max_stages=0, runner=None, workflow_dir=None):
    if min(max_wall_seconds, samples, eval_batch_size, review_bytes, diagnostic_bytes) <= 0 or max_stages < 0:
        raise ValueError("Require positive wall time/panel/intervals and nonnegative max_stages")
    if diagnostic_bytes >= review_bytes:
        raise ValueError("First diagnostic must precede the regular review interval")
    config_path = Path(config_path).resolve()
    cfg = load_config(config_path)
    stable_run = (ROOT / cfg["train"]["run_dir"]).resolve()
    stable_last = stable_run / "last.pt"
    # Keep orchestration metadata outside a fresh trainer's deliberately empty run.
    work = (ROOT / workflow_dir).resolve() if workflow_dir else stable_run.with_name(stable_run.name + "_workflow")
    if work == stable_run:
        raise ValueError("Workflow metadata must be outside the trainer run")
    work.mkdir(parents=True, exist_ok=True)
    state_path = work / "workflow_state.json"
    protocol = {"config_sha256": sha256(config_path), "stable_run": str(stable_run),
                "calibration": str((ROOT / calibration).resolve()), "samples": samples,
                "review_bytes": review_bytes, "diagnostic_bytes": diagnostic_bytes,
                "cooldown_milestones": "review_bytes * powers_of_two"}
    if state_path.is_file():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state["protocol"] != protocol:
            raise ValueError("Workflow restart requires its original model/data/review protocol")
    else:
        previous = progress(stable_last)
        seen = previous["seen_bytes"] if previous else 0
        first_target = max(review_bytes, seen // review_bytes * review_bytes) if seen >= review_bytes else diagnostic_bytes
        state = {"version": 1, "protocol": protocol, "created_at": now(), "completed": [],
                 "pending": {"target_bytes": first_target}, "job_ids": []}
    job = os.environ.get("SLURM_JOB_ID")
    if job and job not in state["job_ids"]:
        state["job_ids"].append(job)
    state.update(status="running", updated_at=now(), stop_reason=None)
    write_json(state_path, state)
    launched = runner or ChildRunner(max_wall_seconds)
    completed_here = 0

    def record(activity):
        state.update(activity=activity, updated_at=now())
        write_json(state_path, state)
        print(json.dumps({"event": "workflow", "activity": activity,
                          "target_bytes": state["pending"]["target_bytes"], "at": now()}), flush=True)

    def invoke_train(training_config, run, *, target=None, source=None):
        command = [sys.executable, "-u", str(SCRIPTS / "train_utf8_ae.py"), str(training_config),
                   "--device", device]
        checkpoint = run / "last.pt"
        if checkpoint.is_file():
            command += ["--resume", str(checkpoint)]
        elif source is not None:
            command += ["--cooldown-from", str(source)]
        if target is not None:
            command += ["--stop-at-bytes", str(target)]
        if not launched.call(command):
            return False
        status = json.loads((run / "status.json").read_text(encoding="utf-8"))
        if status.get("pause_reason") == "signal":
            launched.request_stop("trainer_signal")
            return False
        return True

    def evaluate(checkpoint, output):
        if output.is_file():
            report = json.loads(output.read_text(encoding="utf-8"))
            if report.get("checkpoint_sha256") == sha256(checkpoint):
                return True
        command = [sys.executable, "-u", str(SCRIPTS / "evaluate_utf8_ae.py"),
                   "--checkpoint", str(checkpoint), "--data", str(ROOT / calibration),
                   "--train-data", str(ROOT / cfg["data"]["train"]), "--output", str(output),
                   "--device", device, "--samples", str(samples), "--seed", "1234",
                   "--batch-size", str(eval_batch_size)]
        return launched.call(command)

    try:
        with launched:
            while not launched.stopped():
                pending = state["pending"]
                target = pending["target_bytes"]
                stable = progress(stable_last)
                if stable is not None and stable["phase"] != "stable":
                    raise ValueError("The workflow main run must retain its stable phase")
                if stable is None or stable["seen_bytes"] < target:
                    record("stable_train")
                    before = stable["seen_bytes"] if stable else 0
                    if not invoke_train(config_path, stable_run, target=target):
                        break
                    stable = progress(stable_last)
                    if stable["seen_bytes"] <= before:
                        raise RuntimeError("Stable training returned without making progress")
                    if stable["seen_bytes"] < target:
                        continue  # run_bytes can be smaller than the requested review interval.
                pending["stable_progress"] = stable
                stable_report = work / "evaluations" / f"stable_{target:012d}.json"
                record("stable_evaluate")
                if not evaluate(stable_last, stable_report):
                    break
                pending["stable_evaluation"] = str(stable_report)
                if cooldown_due(target, review_bytes):
                    if stable["step"] < cfg["train"]["warmup_updates"]:
                        pending["cooldown_deferred"] = "warmup still active at this byte milestone"
                    else:
                        branch = stable_run.with_name(stable_run.name + f"_cooldown_{target:012d}")
                        branch_cfg = copy.deepcopy(cfg)
                        branch_cfg["train"]["run_dir"] = str(branch)
                        branch_config = work / "configs" / f"cooldown_{target:012d}.yaml"
                        branch_config.parent.mkdir(parents=True, exist_ok=True)
                        branch_config.write_text(yaml.safe_dump(branch_cfg, sort_keys=False), encoding="utf-8")
                        checkpoint = branch / "last.pt"
                        cooled = progress(checkpoint)
                        while cooled is None or cooled["seen_bytes"] < cooled["decay_start_bytes"] + cooled["decay_budget_bytes"]:
                            record("cooldown_train")
                            if not invoke_train(branch_config, branch, source=stable_last):
                                break
                            cooled = progress(checkpoint)
                        if launched.stopped():
                            break
                        pending["cooldown_progress"] = cooled
                        cooled_report = work / "evaluations" / f"cooldown_{target:012d}.json"
                        record("cooldown_evaluate")
                        if not evaluate(checkpoint, cooled_report):
                            break
                        pending["cooldown_evaluation"] = str(cooled_report)
                state["completed"].append({**pending, "finished_at": now()})
                state["pending"] = {"target_bytes": review_bytes if target < review_bytes else target + review_bytes}
                completed_here += 1
                record("stage_complete")
                if max_stages and completed_here >= max_stages:
                    launched.request_stop("stage_limit")
            state.update(status="paused", stop_reason=launched.stop_reason, updated_at=now(),
                         stable_progress=progress(stable_last))
            write_json(state_path, state)
    except BaseException as exc:
        state.update(status="failed", error=f"{type(exc).__name__}: {exc}", updated_at=now())
        write_json(state_path, state)
        raise
    print(json.dumps({"event": "workflow_paused", "reason": state["stop_reason"],
                      "state": str(state_path), "stable_progress": state.get("stable_progress")}), flush=True)
    return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--calibration", default="data/multilingual_stories/calibration.jsonl")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-wall-seconds", type=float, default=27000)
    parser.add_argument("--samples", type=int, default=4096)
    parser.add_argument("--eval-batch-size", type=int, default=4096)
    parser.add_argument("--review-bytes", type=int, default=512 * MIB)
    parser.add_argument("--diagnostic-bytes", type=int, default=64 * MIB)
    parser.add_argument("--max-stages", type=int, default=0, help="Engineering stop; default runs through the wall-time window")
    parser.add_argument("--workflow-dir", help="Separate orchestration record when deployment-only settings change")
    args = parser.parse_args()
    run_workflow(args.config, calibration=args.calibration, device=args.device,
                 max_wall_seconds=args.max_wall_seconds, samples=args.samples,
                 eval_batch_size=args.eval_batch_size, review_bytes=args.review_bytes,
                 diagnostic_bytes=args.diagnostic_bytes, max_stages=args.max_stages, workflow_dir=args.workflow_dir)


if __name__ == "__main__":
    main()
