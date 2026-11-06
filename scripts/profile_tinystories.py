"""Profile a small TinyStories sample with the existing frozen byte AE.

Downloads metadata and stratified pages of public dataset rows, then measures
story lengths before/after the project's deterministic verify/resplit pipeline.
Run with the cose2 Python environment. This script performs inference only.
"""
from __future__ import annotations

import hashlib
import json
import random
import sys
import time
import urllib.parse
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "external" / "char_lm"  # Optional legacy profiling source
CACHE = ROOT / "data" / "tinystories_probe"
OUT = ROOT / "docs" / "evidence" / "tinystories-profile.json"
DATASET = "roneneldan/TinyStories"
SEED = 20261029


def fetch_json(url):
    req = urllib.request.Request(url, headers={"User-Agent": "char-diff-research/0.1"})
    with urllib.request.urlopen(req, timeout=30) as response:
        return json.load(response)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def describe(values):
    a = np.asarray(values, dtype=np.float64)
    return {"n": len(a), "sum": int(a.sum()), "mean": float(a.mean()),
            **{f"p{q}": float(np.percentile(a, q)) for q in [50, 90, 95, 99]},
            "max": int(a.max())}


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    CACHE.mkdir(parents=True, exist_ok=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    repo = fetch_json(f"https://huggingface.co/api/datasets/{DATASET}")
    size = fetch_json("https://datasets-server.huggingface.co/size?" +
                      urllib.parse.urlencode({"dataset": DATASET}))
    rows_total = {s["split"]: s["num_rows"] for s in size["size"]["splits"]}
    metadata = {"repo_sha": repo["sha"], "card": repo.get("cardData"), "size": size}
    meta_path = CACHE / "metadata.json"
    if meta_path.exists():
        assert json.loads(meta_path.read_text(encoding="utf-8"))["repo_sha"] == repo["sha"]
    meta_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    samples, offsets = {}, {}
    for split, n_pages in [("train", 20), ("validation", 5)]:
        rng = random.Random(SEED + (split == "validation"))
        n = rows_total[split]
        page_offsets = [rng.randrange(i*n//n_pages, (i+1)*n//n_pages-99)
                        for i in range(n_pages)]
        offsets[split] = page_offsets
        path = CACHE / f"{split}.sample.jsonl"
        if path.exists():
            samples[split] = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
            expected = {j for off in page_offsets for j in range(off, off+100)}
            assert {r["row_idx"] for r in samples[split]} == expected
            continue

        def page(off):
            url = "https://datasets-server.huggingface.co/rows?" + urllib.parse.urlencode(
                {"dataset": DATASET, "config": "default", "split": split,
                 "offset": off, "length": 100})
            result = fetch_json(url)
            assert result["num_rows_total"] == n
            assert len(result["rows"]) == 100
            return [{"row_idx": r["row_idx"], "text": r["row"]["text"]} for r in result["rows"]]

        collected = []
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = {pool.submit(page, off): off for off in page_offsets}
            for i, future in enumerate(as_completed(futures), 1):
                collected.extend(future.result())
                print(f"fetch {split}: {i}/{n_pages} pages", flush=True)
        samples[split] = sorted(collected, key=lambda row: row["row_idx"])
        path.write_text("".join(json.dumps(r, ensure_ascii=False)+"\n" for r in samples[split]), encoding="utf-8")

    sys.path.insert(0, str(SOURCE))
    import torch
    from bltz.config import Cfg
    from bltz.models.autoencoder import ByteStringAE
    from bltz.segment_v2 import segment_word
    from bltz.verify import VerifyResplitter

    torch.set_num_threads(4)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint = SOURCE / "checkpoints/ae_48_256_word_full/best.pt"
    tokenizer = SOURCE / "data/qwen35_tokenizer"
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    ae = ByteStringAE(Cfg(state["cfg"]))
    ae.load_state_dict(state["model"], strict=True)
    ae.to(device).eval()
    verifier = VerifyResplitter(ae, str(tokenizer), ent_thresh=0.2, floor_bytes=3,
                              device=device, log_every=0)
    result = {"profile_date": "2026-10-29", "dataset": DATASET, "config": "default",
              "seed": SEED, "sampling": "20 train / 5 validation strata, 100 consecutive rows per stratum",
              "offsets": offsets, "metadata": metadata, "device": device,
              "checkpoint": str(checkpoint), "checkpoint_sha256": digest(checkpoint),
              "checkpoint_step": state.get("step"),
              "ae_parameters": sum(p.numel() for p in ae.parameters()),
              "runtime": {"python": sys.version, "numpy": np.__version__,
                          "torch": torch.__version__,
                          "gpu": torch.cuda.get_device_name() if device == "cuda" else None},
              "source_sha256": {str(p.relative_to(SOURCE)): digest(p) for p in
                                [SOURCE / "bltz/segment_v2.py", SOURCE / "bltz/verify.py",
                                 SOURCE / "bltz/models/autoencoder.py"]},
              "tokenizer_sha256": digest(tokenizer / "tokenizer.json"),
              "verify": {"entropy_bits": 0.2, "floor_bytes": 3, "precision": "float32"},
              "splits": {}}
    for split, records in samples.items():
        raw = [r["text"].encode("utf-8") for r in records]
        segmented = [segment_word(r, str(tokenizer), l_max=32, merge="prev") for r in raw]
        verified = []
        for i in range(0, len(segmented), 64):
            verified.extend(verifier.process_batch(segmented[i:i+64]))
            if i % 512 == 0:
                print(f"verify {split}: {min(i+64,len(segmented))}/{len(segmented)} stories", flush=True)
        assert all(b"".join(u) == r for u, r in zip(verified, raw))
        freq = Counter(u for story in verified for u in story)
        items = list(freq)
        correct_unique, correct_occurrence = 0, 0
        for i in range(0, len(items), 512):
            part = items[i:i+512]
            ok, _ = verifier._verify(part)
            correct_unique += int(ok.sum())
            correct_occurrence += sum(freq[u] for u, good in zip(part, ok) if good)
        lengths = [len(s) for s in verified]
        before = sum(map(len, segmented))
        after = sum(lengths)
        projected = rows_total[split] * float(np.mean(lengths))
        detail = {"raw_utf8_bytes": describe([len(r) for r in raw]),
                  "units_before_verify": describe([len(s) for s in segmented]),
                  "units_after_verify": describe(lengths),
                  "unique_verified_units": len(freq), "unit_inflation": after / before,
                  "ae_exact_occurrence": correct_occurrence / after,
                  "ae_exact_unique": correct_unique / len(freq),
                  "fraction_stories_le_units": {str(c): sum(l<=c for l in lengths)/len(lengths) for c in [128,256,384,512]},
                  "unit_slots_at_block16": sum(((l+15)//16)*16 for l in lengths),
                  "projected_total_units": projected,
                  "sample_sha256": digest(CACHE / f"{split}.sample.jsonl")}
        result["splits"][split] = detail
        print(split, json.dumps(detail, ensure_ascii=False), flush=True)
    assert fetch_json(f"https://huggingface.co/api/datasets/{DATASET}")["sha"] == repo["sha"]
    result["elapsed_seconds"] = time.perf_counter()-started
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print("saved", OUT, flush=True)


if __name__ == "__main__":
    main()
