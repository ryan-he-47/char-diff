"""Plan one UTF-8 document using exact AE reconstruction and quality costs."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch

from char_diff.ae_paths import score_candidates, shortest_path
from char_diff.ae_quality import AEQualityScorer
from char_diff.vendor.bltz.config import Cfg
from char_diff.vendor.bltz.models.autoencoder import ByteStringAE


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--text", help="One UTF-8 document")
    source.add_argument("--input", type=Path, help="UTF-8 text file; original bytes are preserved")
    parser.add_argument("--quality-weight", type=float, default=1.0)
    parser.add_argument("--score-kind", choices=("nll", "entropy"), default="nll")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    raw = args.input.read_bytes() if args.input is not None else args.text.encode("utf-8")
    raw.decode("utf-8", errors="strict")
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    ae = ByteStringAE(Cfg(state["cfg"]))
    ae.load_state_dict(state["model"], strict=True)
    ae.eval()
    scorer = AEQualityScorer(ae, quality_kind=args.score_kind, device=args.device,
                             batch_size=args.batch_size)
    # Scores live only for this document; no vectors or persistent candidate cache.
    scores = score_candidates(raw, scorer, max_bytes=ae.l_max, batch_size=args.batch_size)
    result = shortest_path(raw, scores, max_bytes=ae.l_max, quality_weight=args.quality_weight)
    digest = hashlib.sha256()
    with args.checkpoint.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    record = {
        "checkpoint": str(args.checkpoint.resolve()), "checkpoint_sha256": digest.hexdigest(),
        "checkpoint_step": state.get("step"), "model_parameters": sum(p.numel() for p in ae.parameters()),
        "boundary_mode": "utf8", "max_unit_bytes": ae.l_max,
        "score_kind": args.score_kind, "score_units": "nats_including_eos",
        "quality_weight": args.quality_weight, "device": str(scorer.device),
        "precision": "fp32", "batch_size": args.batch_size,
        "input_bytes": len(raw), "unique_candidates": len(scores),
        "unit_count": len(result.units), "cost": result.cost, "quality_sum": result.quality_sum,
        "boundaries": result.boundaries, "byte_conserved": b"".join(result.units) == raw,
        "units": [{"hex": u.hex(), "bytes": len(u), "text": u.decode("utf-8"),
                   "quality": scores[u].quality} for u in result.units],
    }
    payload = json.dumps(record, ensure_ascii=True, indent=2, allow_nan=False) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
        print(json.dumps({"output": str(args.output.resolve()), "unit_count": len(result.units),
                          "byte_conserved": record["byte_conserved"]}))
    else:
        print(payload, end="")


if __name__ == "__main__":
    main()
