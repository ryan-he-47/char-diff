"""Focused real-architecture FP16 resume and invalid-gradient smoke."""
import copy
import json
import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from char_diff import ae_training
from char_diff.utils import ROOT, load_config, write_json


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


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("This smoke checks the CUDA FP16 path")
    out = ROOT / "runs/utf8_ae_build_20261103"
    out.mkdir(parents=True, exist_ok=True)
    fixture = out / "fixture.jsonl"
    texts = {
        "english": "A small cat found a red ball. She smiled!",
        "spanish": "La niña encontró una pelota roja y sonrió.",
        "chinese": "小猫找到一个红色的球。她高兴地笑了！",
        "turkish": "Küçük kedi kırmızı bir top buldu.",
        "arabic": "وجدت القطة الصغيرة كرة حمراء.",
        "vietnamese": "Cô bé tìm thấy một quả bóng đỏ.",
        "azerbaijani": "Balaca pişik qırmızı top tapdı.",
        "korean": "작은 고양이가 빨간 공을 찾았어요.",
        "german": "Die kleine Katze fand einen roten Ball.",
        "farsi": "گربه کوچک یک توپ قرمز پیدا کرد.",
        "hindi": "छोटी बिल्ली को एक लाल गेंद मिली।",
        "hebrew": "החתולה הקטנה מצאה כדור אדום.",
    }
    with fixture.open("w", encoding="utf-8") as f:
        for language, text in texts.items():
            f.write(json.dumps({"id": language, "language": language, "split": "train", "text": text}, ensure_ascii=False) + "\n")
    base = load_config(ROOT / "configs/tinystories_multilingual_ae48.yaml")
    base["data"].update(train=str(fixture), manifest=None)
    base["train"].update(batch_windows=16, micro_batch=8, log_every=1, save_every=2,
                         warmup_updates=2, run_bytes=4096, diagnostic_bytes=350)

    def config(name):
        cfg = copy.deepcopy(base)
        cfg["train"]["run_dir"] = str(out / name)
        return cfg

    full_cfg, split_cfg = config("continuous"), config("resumed")
    full = ae_training.train(full_cfg, stop_after=3, device="cuda")
    ae_training.train(split_cfg, stop_after=2, device="cuda")
    restored = ae_training.train(split_cfg, resume=str(out / "resumed/last.pt"), stop_after=1, device="cuda")
    a = torch.load(out / "continuous/last.pt", map_location="cpu", weights_only=False)
    b = torch.load(out / "resumed/last.pt", map_location="cpu", weights_only=False)
    for key in ("model", "optimizer", "scaler", "state", "sampler", "rng"):
        assert equal(a[key], b[key]), key
    assert full["parameters"] == 3015537 and full["skipped"] == restored["skipped"] == 0
    assert (out / "resumed/previous.pt").is_file()
    assert (out / "resumed/diagnostic.pt").is_file()
    original_ce = ae_training.F.cross_entropy
    calls = [0]

    def inject_once(*args, **kwargs):
        loss = original_ce(*args, **kwargs)
        calls[0] += 1
        return loss * float("nan") if calls[0] == 1 else loss

    with patch.object(ae_training.F, "cross_entropy", inject_once):
        protected = ae_training.train(config("injected_invalid"), stop_after=1, device="cuda")
    assert protected["skipped"] == 1 and protected["attempts"] == 2 and protected["step"] == 1
    invalid = torch.load(out / "injected_invalid/last.pt", map_location="cpu", weights_only=False)
    assert all(torch.isfinite(v).all() for v in invalid["model"].values())
    evidence = {"status": "passed", "device": torch.cuda.get_device_name(0), "parameters": full["parameters"],
                "precision": "fp16", "fixture": "synthetic 12-language strings; no corpus download",
                "resume": "3 continuous vs 2+1: model/optimizer/scaler/state/sampler/RNG bit-identical",
                "normal_skips": 0, "injected_invalid": "1 skipped attempt followed by 1 finite update",
                "config_overrides": {k: base["train"][k] for k in ("batch_windows", "micro_batch", "warmup_updates", "run_bytes", "diagnostic_bytes", "save_every")},
                "continuous": full, "resumed": restored, "protection": protected}
    write_json(out / "evidence.json", evidence)
    print("PASS: FP16 exact resume and invalid-gradient protection")


if __name__ == "__main__":
    main()
