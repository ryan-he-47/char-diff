"""Focused evaluation aggregation, paired-panel determinism and real-AE smoke."""
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from char_diff.ae_windows import UTF8WindowSampler
from char_diff.utils import ROOT, write_json
from scripts.evaluate_utf8_ae import aggregate, build_panels, evaluate, panel_fingerprint


def main():
    rows = [{"bytes": 1, "target_nll_nats": 2., "feasible": True, "eos_argmax": True},
            {"bytes": 3, "target_nll_nats": 6., "feasible": False, "eos_argmax": True}]
    stats = aggregate(rows)
    assert stats["byte_and_eos_positions"] == 6
    assert stats["nll_per_byte_and_eos"] == 8 / 6 and stats["string_em"] == .5
    out = ROOT / "runs/utf8_ae_wsd_20261103/eval_smoke"
    out.mkdir(parents=True, exist_ok=True)
    fixture = out / "fixture.jsonl"
    records = [
        {"id": "cal-en", "language": "english", "split": "calibration", "text": "A cat found a small red ball."},
        {"id": "cal-zh", "language": "chinese", "split": "calibration", "text": "小猫发现了一个红色的球。"},
        {"id": "train-en", "language": "english", "split": "train", "text": "The little cat is happy."},
        {"id": "train-zh", "language": "chinese", "split": "train", "text": "小猫非常开心。"},
    ]
    fixture.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records), encoding="utf-8")
    fingerprints = []
    for _ in range(2):
        with UTF8WindowSampler(fixture, split="calibration", seed=1234, augment_probability=0.,
                               single_character_probability=0.) as calibration, \
             UTF8WindowSampler(fixture, split="train", seed=1235) as training:
            natural, augmented = build_panels(calibration, training, 16)
            assert all(a.original == b.original and a.language == b.language for a, b in zip(natural, augmented))
            assert all(len(s.data) <= 32 and s.data.decode("utf-8") for s in natural + augmented)
            fingerprints.append([panel_fingerprint(natural), panel_fingerprint(augmented)])
    assert fingerprints[0] == fingerprints[1]
    result = evaluate(ROOT / "assets/ae_best.pt", fixture, out / "report.json", train_data=fixture,
                      samples=32, device="cpu", batch_size=16)
    assert [result["panels"][name]["samples_sha256"] for name in ("natural", "augmented")] == fingerprints[0]
    for panel in result["panels"].values():
        assert panel["overall"]["strings"] == 16
        assert sum(row["strings"] for row in panel["by_language"].values()) == 16
        assert sum(row["byte_and_eos_positions"] for row in panel["by_byte_length"].values()) == panel["overall"]["byte_and_eos_positions"]
    assert "train_observed_characters" in result["character_coverage"]
    assert result["character_coverage"]["calibration_characters_absent_from_training"]["strings"] > 0
    write_json(out / "evidence.json", {"status": "passed", "checks": [
        "byte+EOS position-weighted NLL and complete-string EM",
        "paired targets and deterministic fingerprints",
        "real old AE FP32 CPU on 32 fixture windows plus observed characters; group totals and coverage labels"],
        "checkpoint": "assets/ae_best.pt", "formal_data": False})
    print("PASS: AE evaluation aggregation, fixed pairs and real-checkpoint smoke")


if __name__ == "__main__":
    main()
