"""Focused WSD schedule, bounded checkpoints and full-state FP16 resume smoke."""
import contextlib
import copy
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from char_diff.ae_training import learning_rate, train
from char_diff.utils import ROOT, load_config, sha256, write_json


def equal(a, b):
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and torch.equal(a, b)
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(equal(a[k], b[k]) for k in a)
    if isinstance(a, (tuple, list)):
        return len(a) == len(b) and all(equal(x, y) for x, y in zip(a, b))
    if hasattr(a, "shape"):
        return bool((a == b).all())
    return a == b


def load(path):
    return torch.load(path, map_location="cpu", weights_only=False)


def compare(path_a, path_b):
    a, b = load(path_a), load(path_b)
    for key in ("model", "optimizer", "scaler", "state", "sampler", "rng", "origin"):
        assert equal(a[key], b[key]), key
    return a


def test_schedule(tc):
    cfg = dict(tc, warmup_updates=2)
    state = {"phase": "stable", "step": 0, "seen_bytes": 0}
    assert learning_rate(state, cfg) == cfg["lr"] / 2
    state.update(step=1, seen_bytes=80)
    assert learning_rate(state, cfg) == cfg["lr"]
    state.update(step=2, seen_bytes=160)
    assert learning_rate(state, cfg) == cfg["lr"]
    for budget in (1, 1024, 2**40):
        for seen in (0, budget, budget * 10):
            state.update(step=9999, seen_bytes=seen)
            assert learning_rate(state, dict(cfg, run_bytes=budget)) == cfg["lr"]
    state.update(phase="cooldown", seen_bytes=1000, decay_start_bytes=1000, decay_budget_bytes=400)
    assert learning_rate(state, cfg) == cfg["lr"]
    assert math.isclose(learning_rate(state, cfg, update_bytes=200), (cfg["lr"] + cfg["final_lr"]) / 2, rel_tol=1e-12)
    assert learning_rate(state, cfg, update_bytes=400) == cfg["final_lr"]
    assert learning_rate(state, cfg, update_bytes=800) == cfg["final_lr"]
    # Failed updates leave seen_bytes unchanged, so asking for the same next
    # batch returns exactly the same decay LR.
    assert learning_rate(state, cfg, update_bytes=80) == learning_rate(dict(state), cfg, update_bytes=80)


def compact(result):
    keys = ("event", "pause_reason", "phase", "step", "attempts", "seen_bytes", "skipped",
            "decay_start_bytes", "decay_budget_bytes", "stage_checkpoints", "parameters")
    return {k: result[k] for k in keys}


def assert_retention(run, expected):
    assert sorted(p.name for p in run.glob("stage_*.pt")) == sorted(expected)
    assert len(expected) == 2
    assert (run / "last.pt").is_file()
    assert (run / "previous.pt").is_file()


def verify_endpoints_and_source():
    """Recheck only the final endpoint/hash-load adjustment, without retraining."""
    out = ROOT / "runs/utf8_ae_wsd_20261103/training_smoke"
    source = out / "stable_continuous/last.pt"
    before = sha256(source)
    original = load(source)
    cfg = copy.deepcopy(original["config"])
    test_schedule(cfg["train"])
    cfg["train"]["run_dir"] = str(out / "source_load_probe")
    with (out / "source_load_probe.log").open("w", encoding="utf-8") as log, contextlib.redirect_stdout(log):
        result = train(cfg, cooldown_from=str(source), stop_at_bytes=original["state"]["seen_bytes"], device="cuda")
    loaded = load(out / "source_load_probe/last.pt")
    for key in ("model", "optimizer", "scaler", "sampler", "rng"):
        assert equal(original[key], loaded[key]), key
    for key in ("step", "attempts", "seen_bytes", "attempted_bytes", "skipped"):
        assert original["state"][key] == loaded["state"][key], key
    assert loaded["origin"]["sha256"] == before == sha256(source)
    assert result["pause_reason"] == "stop_at_bytes"
    evidence_path = out / "evidence.json"
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    evidence["endpoint_source_followup"] = {
        "status": "passed", "new_training_updates": 0,
        "precise_lr_endpoints": "peak and floor exactly equal configured floats",
        "same_handle_source_load": "full payload unchanged, source digest matches before/after",
        "trainer_sha256": sha256(ROOT / "char_diff/ae_training.py"),
    }
    write_json(evidence_path, evidence)
    print("PASS: exact LR endpoints and same-handle source load, zero extra training updates")


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("This smoke exercises CUDA FP16")
    out = ROOT / "runs/utf8_ae_wsd_20261103/training_smoke"
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Preserve existing smoke evidence: {out}")
    out.mkdir(parents=True, exist_ok=True)
    fixture = ROOT / "runs/utf8_ae_build_20261103/fixture.jsonl"
    if not fixture.is_file():
        raise FileNotFoundError("Expected the existing synthetic multilingual fixture")
    base = load_config(ROOT / "configs/tinystories_multilingual_ae48.yaml")
    test_schedule(base["train"])
    base["data"].update(train=str(fixture), manifest=None)
    base["train"].update(batch_windows=8, micro_batch=4, warmup_updates=2, run_bytes=100000,
                         cooldown_bytes=400, stage_every_bytes=80, keep_stages=2,
                         diagnostic_bytes=64, save_every=2, log_every=1)

    def config(name):
        cfg = copy.deepcopy(base)
        cfg["train"]["run_dir"] = str(out / name)
        return cfg

    with (out / "smoke.log").open("w", encoding="utf-8") as log, contextlib.redirect_stdout(log):
        continuous = train(config("stable_continuous"), stop_after=4, device="cuda")
        split_cfg = config("stable_resumed")
        split_cfg["train"]["run_bytes"] = 1
        paused = train(split_cfg, device="cuda")
        assert paused["pause_reason"] == "run_bytes" and paused["step"] == 1
        split_cfg["train"]["run_bytes"] = 100000
        resumed = train(split_cfg, resume=str(out / "stable_resumed/last.pt"), stop_after=3, device="cuda")
        stable = compare(out / "stable_continuous/last.pt", out / "stable_resumed/last.pt")
        assert stable["state"]["step"] == 4 and stable["state"]["skipped"] == 0
        assert continuous["parameters"] == 3015537
        for run_name in ("stable_continuous", "stable_resumed"):
            run = out / run_name
            assert_retention(run, stable["state"]["stage_checkpoints"])
            assert (run / "diagnostic.pt").is_file()
        for group in stable["optimizer"]["param_groups"]:
            assert group["lr"] == base["train"]["lr"]

        source = out / "stable_continuous/last.pt"
        source_hash = sha256(source)
        stable_stage_hashes = {p.name: sha256(p) for p in source.parent.glob("stage_*.pt")}
        cooled = train(config("cooldown_continuous"), cooldown_from=str(source), device="cuda")
        cooldown_cfg = config("cooldown_resumed")
        mid = train(cooldown_cfg, cooldown_from=str(source), stop_after=2, device="cuda")
        assert mid["event"] == "paused" and mid["phase"] == "cooldown"
        cooldown_cfg["train"]["run_bytes"] = 200000
        cooled_resumed = train(cooldown_cfg, resume=str(out / "cooldown_resumed/last.pt"), device="cuda")
        decay = compare(out / "cooldown_continuous/last.pt", out / "cooldown_resumed/last.pt")
        assert cooled["event"] == cooled_resumed["event"] == "complete"
        assert decay["state"]["skipped"] == 0
        assert decay["state"]["seen_bytes"] >= decay["state"]["decay_start_bytes"] + 400
        for group in decay["optimizer"]["param_groups"]:
            assert math.isclose(group["lr"], base["train"]["final_lr"], rel_tol=1e-12)
        for run_name in ("cooldown_continuous", "cooldown_resumed"):
            assert_retention(out / run_name, decay["state"]["stage_checkpoints"])
        assert sha256(source) == source_hash
        assert {p.name: sha256(p) for p in source.parent.glob("stage_*.pt")} == stable_stage_hashes

    evidence = {
        "status": "passed", "device": torch.cuda.get_device_name(0), "precision": "fp16",
        "parameters": continuous["parameters"], "fixture_sha256": sha256(fixture),
        "fixture_scope": "Existing synthetic 12-language fixture; no corpus download",
        "schedule": "warmup/stable and linear cooldown endpoints passed; independent run_bytes",
        "stable_resume": "4 continuous vs 1 budget-pause + 3 resumed: model/optimizer/scaler/state/sampler/RNG/origin bit-identical",
        "cooldown_resume": "same source, continuous vs 2-step pause+resume: full state bit-identical",
        "checkpoint_retention": "2 stage files, last/previous, stable diagnostic preserved; source hashes unchanged",
        "config_overrides": {k: base["train"][k] for k in ("batch_windows", "micro_batch", "warmup_updates", "run_bytes", "cooldown_bytes", "stage_every_bytes", "keep_stages", "diagnostic_bytes", "save_every")},
        "stable": compact(continuous), "quota_pause": compact(paused), "resumed": compact(resumed),
        "cooldown": compact(cooled), "cooldown_pause": compact(mid), "cooldown_resumed": compact(cooled_resumed),
        "cooldown_final_lr": decay["optimizer"]["param_groups"][0]["lr"],
        "source_checkpoint_sha256": source_hash,
    }
    write_json(out / "evidence.json", evidence)
    print(json.dumps({"status": "passed", "stable_steps": continuous["step"],
                      "stable_bytes": continuous["seen_bytes"], "cooldown_new_steps": cooled["step"] - continuous["step"],
                      "cooldown_new_bytes": cooled["seen_bytes"] - continuous["seen_bytes"],
                      "normal_skips": 0}, ensure_ascii=False))


if __name__ == "__main__":
    if "--verify-endpoints-and-source" in sys.argv:
        verify_endpoints_and_source()
    else:
        main()
