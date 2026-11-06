"""Fixed UTF-8 reconstruction panels for comparing AE training checkpoints."""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch

from char_diff.ae_quality import AEQualityScorer
from char_diff.ae_windows import UTF8WindowSampler, character_replacement_probability
from char_diff.utils import write_json
from char_diff.vendor.bltz.config import Cfg
from char_diff.vendor.bltz.models.autoencoder import ByteStringAE


def length_bucket(n):
    low = 8 * ((n - 1) // 8) + 1
    return f"{low}-{low + 7}"


def aggregate(records):
    """NLL weights supervised positions; EM weights complete strings equally."""
    count = len(records)
    positions = sum(row["bytes"] + 1 for row in records)
    nll = sum(row["target_nll_nats"] for row in records)
    exact = sum(row["feasible"] for row in records)
    eos = sum(row["eos_argmax"] for row in records)
    return {"strings": count, "target_bytes": positions - count,
            "byte_and_eos_positions": positions, "nll_sum_nats": nll,
            "nll_per_byte_and_eos": nll / positions if positions else None,
            "exact_strings": exact, "string_em": exact / count if count else None,
            "correct_eos": eos, "eos_accuracy": eos / count if count else None}


def grouped(samples, records):
    by_language, by_length, joint = {}, {}, {}
    for sample, row in zip(samples, records):
        length = length_bucket(row["bytes"])
        by_language.setdefault(sample.language, []).append(row)
        by_length.setdefault(length, []).append(row)
        joint.setdefault(sample.language, {}).setdefault(length, []).append(row)
    return {"overall": aggregate(records),
            "by_language": {key: aggregate(value) for key, value in sorted(by_language.items())},
            "by_byte_length": {key: aggregate(value) for key, value in sorted(by_length.items())},
            "by_language_and_byte_length": {
                lang: {key: aggregate(value) for key, value in sorted(groups.items())}
                for lang, groups in sorted(joint.items())}}


def panel_fingerprint(samples):
    digest = hashlib.sha256()
    for sample in samples:
        row = [sample.language, sample.doc_id, sample.start_byte, sample.end_byte,
               sample.original.hex(), sample.data.hex()]
        digest.update(json.dumps(row, ensure_ascii=False, separators=(",", ":")).encode("utf-8") + b"\n")
    return digest.hexdigest()


def build_panels(calibration, replacement_source, count):
    """Paired targets; augmentation does not alter which natural windows we draw.

    Held-out source characters may be absent from the replacement inventory;
    then every inventory entry is a different character and remains eligible.
    """
    natural = calibration.sample_batch(count)
    augmented = []
    inventory = replacement_source.characters
    indices = {char: i for i, char in enumerate(inventory)}
    rng = replacement_source.rng
    for sample in natural:
        chars = list(sample.data.decode("utf-8"))
        size, accepted, rejected = len(sample.data), 0, 0
        for i, char in enumerate(chars):
            width = len(char.encode("utf-8"))
            counts = replacement_source.stats["replacement_by_width"][str(width)]
            counts["characters"] += 1
            counts["source_bytes"] += width
            if rng.random() >= character_replacement_probability(replacement_source.replace_probability, width):
                continue
            counts["triggered"] += 1
            old_index = indices.get(char)
            candidates = len(inventory) - int(old_index is not None)
            if not candidates:
                counts["no_alternative"] += 1
                continue
            pick = rng.randrange(candidates)
            if old_index is not None:
                pick += pick >= old_index
            new_char = inventory[pick]
            new_size = size - width + len(new_char.encode("utf-8"))
            if new_size > replacement_source.max_bytes:
                rejected += 1
                counts["rejected_length"] += 1
                continue
            chars[i] = new_char
            size = new_size
            accepted += 1
            counts["accepted"] += 1
        augmented.append(replace(sample, data="".join(chars).encode("utf-8"), augmentation_selected=True,
                                 substitutions=accepted, rejected_substitutions=rejected))
    return natural, augmented


def character_report(sampler, scores):
    records = [scores[char] for char in sampler.characters]
    widths = {}
    for row in records:
        widths.setdefault(str(row["bytes"]), []).append(row)
    return {"definition": "unique source codepoints reconstructed as a complete byte+EOS string",
            "overall": aggregate(records),
            "by_utf8_width": {key: aggregate(value) for key, value in sorted(widths.items())},
            "by_language": {
                lang: aggregate([scores[char] for char in chars])
                for lang, chars in sorted(sampler._language_characters.items())}}


def evaluate(checkpoint, data, output, *, train_data=None, split="calibration",
             device="cpu", samples=4096, seed=1234, batch_size=512, replace_probability=None):
    if samples < 2 or samples % 2:
        raise ValueError("samples must be positive and even; half per paired panel")
    torch.set_num_threads(4)
    checkpoint, data, output = Path(checkpoint), Path(data), Path(output)
    # Pin one open checkpoint across hashing/loading while last.pt is rotated.
    with checkpoint.open("rb") as handle:
        digest = hashlib.sha256()
        for chunk in iter(lambda: handle.read(8 * 1024**2), b""):
            digest.update(chunk)
        handle.seek(0)
        saved = torch.load(handle, map_location="cpu", weights_only=False)
    checkpoint_sha256 = digest.hexdigest()
    ae = ByteStringAE(Cfg(saved["cfg"]))
    ae.load_state_dict(saved["model"], strict=True)
    scorer = AEQualityScorer(ae, device=device, batch_size=batch_size)
    if replace_probability is None:
        replace_probability = saved.get("config", {}).get("data", {}).get("replace_probability", .05)
    with ExitStack() as stack:
        calibration = stack.enter_context(UTF8WindowSampler(
            data, split=split, seed=seed, max_bytes=ae.l_max,
            preload_texts=True, cache_documents=4096,
            augment_probability=0., single_character_probability=0.,
            replace_probability=replace_probability))
        replacement_source = calibration
        training = None
        if train_data is not None:
            training = stack.enter_context(UTF8WindowSampler(
                train_data, split="train", seed=seed + 1, max_bytes=ae.l_max,
                preload_texts=True,
                augment_probability=0., single_character_probability=0.,
                replace_probability=replace_probability))
            replacement_source = training
        natural, augmented = build_panels(calibration, replacement_source, samples // 2)
        panels = {}
        for name, panel in (("natural", natural), ("augmented", augmented)):
            records = scorer.score_records([sample.data for sample in panel])
            panels[name] = {"samples_sha256": panel_fingerprint(panel), **grouped(panel, records)}
        characters = sorted(set(calibration.characters) | (set(training.characters) if training else set()))
        scores = dict(zip(characters, scorer.score_records([char.encode("utf-8") for char in characters])))
        char_reports = {f"{split}_observed_characters": character_report(calibration, scores)}
        if training:
            char_reports["train_observed_characters"] = character_report(training, scores)
            unseen = sorted(set(calibration.characters) - set(training.characters))
            char_reports[f"{split}_characters_absent_from_training"] = aggregate([scores[c] for c in unseen])
        result = {
            "format_version": 1, "checkpoint": str(checkpoint.resolve()),
            "checkpoint_sha256": checkpoint_sha256, "checkpoint_step": saved.get("step"),
            "training_state": saved.get("state"), "precision": "fp32", "device": str(device),
            "parameters": sum(p.numel() for p in ae.parameters()),
            "protocol": {"seed": seed, "samples_total": samples, "paired_panels": True,
                         "split": split, "data": str(data.resolve()),
                         "source_sha256": calibration.source_sha256,
                         "train_source_sha256": training.source_sha256 if training else None,
                         "replacement_inventory": "train" if training else split,
                         "replace_probability": replace_probability,
                         "replacement_probability_basis": "per_original_byte_union_per_codepoint",
                         "augmented_panel_gate_probability": 1.0,
                         "single_character_practice_probability": 0.0,
                         "sampling": calibration.protocol["sampling"],
                         "character_coverage": "all unique observed codepoints, no sampling",
                         "aggregation": "NLL / total byte+EOS positions; EM / complete strings"},
            "augmentation": {"changed_strings": sum(s.data != s.original for s in augmented),
                             "accepted_substitutions": sum(s.substitutions for s in augmented),
                             "rejected_length": sum(s.rejected_substitutions for s in augmented),
                             "by_original_utf8_width": replacement_source.stats["replacement_by_width"]},
            "panels": panels, "character_coverage": char_reports,
        }
    write_json(output, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True, help="Immutable calibration JSONL")
    parser.add_argument("--train-data", type=Path, help="Training JSONL for replacement inventory and character coverage")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", default="calibration")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--samples", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--replace-probability", type=float)
    args = parser.parse_args()
    result = evaluate(**vars(args))
    print(json.dumps({"output": str(args.output.resolve()), "checkpoint_step": result["checkpoint_step"],
                      "panels": {name: panel["overall"] for name, panel in result["panels"].items()}},
                     ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
