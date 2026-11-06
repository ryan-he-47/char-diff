"""Stream a small, language-balanced TinyStories pool without a latent cache.

Example (network preparation; run only where data preparation is authorized)::

    python scripts/prepare_multilingual_stories.py --output data/multilingual_stories

Offline fixtures are JSONL rows with ``language`` (a config name) and ``text``.
``source_config`` and ``story`` are accepted aliases. Text is preserved exactly.
Only ``manifest.json`` with status ``complete`` admits the resulting pool.
"""

import argparse
import hashlib
import json
import os
from contextlib import ExitStack
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation, ROUND_CEILING
from pathlib import Path


REPO = "Gabrui/multilingual_TinyStories"
REVISION = "7fb5f327b907318dd95877a366e288f1971b64c1"
LANGUAGES = (
    "english", "spanish", "chinese", "turkish", "arabic", "vietnamese",
    "azerbaijani", "korean", "german", "farsi", "hindi", "hebrew",
)
LANGUAGE_CODES = dict(zip(LANGUAGES, (
    "en", "es", "zh", "tr", "ar", "vi", "az", "ko", "de", "fa", "hi", "he",
)))
SPLITS = ("train", "calibration", "test")
MIB = 1024 * 1024
TRANSLATION_CAVEAT = (
    "Splits isolate exact UTF-8 document content globally by SHA256. "
    "The upstream story-only schema has no translation-group or original-story ID; "
    "translations of the same source story may occur in different splits."
)


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(MIB), b""):
            digest.update(chunk)
    return digest.hexdigest()


def split_for_hash(content_sha256, seed):
    """A fixed 32:1:1 content-hash split, independent of language and pool size."""
    payload = f"multilingual-stories-v1\n{seed}\n{content_sha256}".encode("ascii")
    bucket = int.from_bytes(hashlib.sha256(payload).digest(), "big") % 34
    return "train" if bucket < 32 else ("calibration" if bucket == 32 else "test")


def mib_value(value):
    try:
        number = Decimal(value)
    except InvalidOperation as exc:
        raise argparse.ArgumentTypeError("MiB must be a positive finite number") from exc
    if not number.is_finite() or number <= 0:
        raise argparse.ArgumentTypeError("MiB must be a positive finite number")
    return number


def byte_quota(mib, languages=1):
    return int((mib * MIB / languages).to_integral_value(rounding=ROUND_CEILING))


def fixture_rows(path, language):
    """Rescan a local fixture per language, keeping memory bounded like streaming."""
    source_row = 0
    with Path(path).open("r", encoding="utf-8", errors="strict") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            config = row.get("source_config", row.get("language"))
            if config not in LANGUAGES:
                raise ValueError(f"Fixture line {line_number}: unknown language {config!r}")
            if config == language:
                yield source_row, row.get("text", row.get("story"))
                source_row += 1


def dataset_rows(language):
    # Lazy import keeps the fixture route independent of Hugging Face dependencies.
    from datasets import load_dataset

    stream = load_dataset(
        REPO, language, revision=REVISION, split="train", streaming=True,
    )
    for source_row, row in enumerate(stream):
        yield source_row, row["story"]


def write_manifest(output, manifest):
    temporary = output / ".manifest.json.partial"
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(output / "manifest.json")


def prepare(args):
    languages = [part for value in args.languages for part in value.split(",")]
    if not languages or len(set(languages)) != len(languages):
        raise ValueError("Select distinct language config names")
    unknown = set(languages) - set(LANGUAGES)
    if unknown:
        raise ValueError(f"Unknown language configs: {sorted(unknown)}")
    quotas = {
        "train": byte_quota(args.train_mib, len(languages)),
        "calibration": byte_quota(args.eval_mib, len(languages)),
        "test": byte_quota(args.eval_mib, len(languages)),
    }
    max_raw_bytes = byte_quota(args.max_raw_mib)
    if max_raw_bytes > 160 * MIB:
        raise ValueError("This prototype is capped at 160 MiB of raw text")
    if sum(quotas.values()) * len(languages) > max_raw_bytes:
        raise ValueError("Requested raw text quotas exceed --max-raw-mib")
    fixture = Path(args.fixture).resolve() if args.fixture else None
    if fixture and not fixture.is_file():
        raise FileNotFoundError(fixture)
    fixture_metadata = ({"path": str(fixture), "sha256": file_sha256(fixture)}
                        if fixture else None)
    output = Path(args.output).resolve()
    if output.exists():
        if not output.is_dir() or any(output.iterdir()):
            raise FileExistsError(f"Refusing to overwrite nonempty output: {output}")
    else:
        output.mkdir(parents=True)
    # Claim the empty directory before opening any output. A second writer fails.
    lock = output / ".preparation.lock"
    with lock.open("x", encoding="ascii") as handle:
        handle.write(str(os.getpid()))

    counts = {
        lang: {
            "scanned_documents": 0, "empty_documents": 0,
            "duplicate_documents": 0, "full_split_documents": 0,
            "splits": {split: {"documents": 0, "raw_text_bytes": 0} for split in SPLITS},
        } for lang in languages
    }
    manifest = {
        "format_version": 1, "status": "preparing", "created_at": utc_now(),
        "dataset": {"repo": REPO, "revision": REVISION, "upstream_split": "train",
                    "license": "cdla-sharing-1.0", "text_field": "story"},
        "source_mode": "fixture" if fixture else "huggingface_streaming",
        "fixture": fixture_metadata,
        "languages": languages,
        "language_codes": {lang: LANGUAGE_CODES[lang] for lang in languages},
        "seed": args.seed,
        "split_assignment": {
            "algorithm": "SHA256(ASCII('multilingual-stories-v1\\n' + seed + '\\n' + content_sha256)); unsigned big-endian integer modulo 34",
            "buckets": {"train": "0..31", "calibration": "32", "test": "33"},
            "independent_of_language_and_quota": True,
        },
        "document_id": "id is lowercase SHA256 hex of exact UTF-8 text bytes",
        "deduplication": "global exact content SHA256 across all emitted languages and splits",
        "source_row": "zero-based row index within the selected source config; fixture indices count matching-language rows",
        "text_policy": "strict UTF-8; preserve complete text including whitespace; skip empty text; no normalization or truncation",
        "selection": "source order, one config at a time; stop each split after its first complete document reaches quota",
        "quota": {
            "requested_total_mib": {"train": str(args.train_mib),
                                    "calibration": str(args.eval_mib), "test": str(args.eval_mib)},
            "per_language_raw_text_bytes": quotas,
            "max_total_raw_text_bytes": max_raw_bytes,
            "overshoot": "at most one complete final document per language/split; hard total cap still applies",
            "accounting": "raw UTF-8 text bytes; JSONL metadata/escaping adds separately reported artifact bytes",
        },
        "translation_group_caveat": TRANSLATION_CAVEAT,
        "counts": counts, "total_documents": 0, "total_raw_text_bytes": 0,
        "artifacts": {},
    }
    seen = set()
    partial_paths = {split: output / f".{split}.jsonl.partial" for split in SPLITS}
    digests = {split: hashlib.sha256() for split in SPLITS}
    try:
        write_manifest(output, manifest)
        with ExitStack() as stack:
            files = {split: stack.enter_context(path.open("xb"))
                     for split, path in partial_paths.items()}
            for language in languages:
                stats = counts[language]
                rows = fixture_rows(fixture, language) if fixture else dataset_rows(language)
                for source_row, text in rows:
                    stats["scanned_documents"] += 1
                    if not isinstance(text, str):
                        raise TypeError(f"{language} row {source_row}: text must be a string")
                    raw = text.encode("utf-8", errors="strict")
                    if not raw:
                        stats["empty_documents"] += 1
                        continue
                    content_hash = hashlib.sha256(raw).hexdigest()
                    if content_hash in seen:
                        stats["duplicate_documents"] += 1
                        continue
                    split = split_for_hash(content_hash, args.seed)
                    target = stats["splits"][split]
                    if target["raw_text_bytes"] >= quotas[split]:
                        stats["full_split_documents"] += 1
                        continue
                    if manifest["total_raw_text_bytes"] + len(raw) > max_raw_bytes:
                        raise RuntimeError("Complete-document addition would exceed the raw text byte cap")
                    record = {"id": content_hash, "language": language, "split": split,
                              "text": text, "source_row": source_row, "source_config": language}
                    encoded = (json.dumps(record, ensure_ascii=False, separators=(",", ":"),
                                          allow_nan=False) + "\n").encode("utf-8")
                    files[split].write(encoded)
                    digests[split].update(encoded)
                    seen.add(content_hash)
                    target["documents"] += 1
                    target["raw_text_bytes"] += len(raw)
                    manifest["total_documents"] += 1
                    manifest["total_raw_text_bytes"] += len(raw)
                    if all(stats["splits"][name]["raw_text_bytes"] >= quotas[name]
                           for name in SPLITS):
                        break
                missing = [name for name in SPLITS
                           if stats["splits"][name]["raw_text_bytes"] < quotas[name]]
                if missing:
                    raise RuntimeError(f"{language}: source exhausted before quotas for {', '.join(missing)}")
                write_manifest(output, manifest)
                print(json.dumps({"language": language, "splits": stats["splits"]}), flush=True)
            for handle in files.values():
                handle.flush()
                os.fsync(handle.fileno())
        for split, partial in partial_paths.items():
            final = output / f"{split}.jsonl"
            partial.rename(final)
            manifest["artifacts"][split] = {
                "path": final.name, "bytes": final.stat().st_size,
                "sha256": digests[split].hexdigest(),
                "documents": sum(counts[lang]["splits"][split]["documents"] for lang in languages),
                "raw_text_bytes": sum(counts[lang]["splits"][split]["raw_text_bytes"] for lang in languages),
            }
        manifest["total_artifact_bytes"] = sum(item["bytes"] for item in manifest["artifacts"].values())
        manifest["status"] = "complete"
        manifest["completed_at"] = utc_now()
        write_manifest(output, manifest)
    except BaseException as exc:
        manifest["status"] = "failed"
        manifest["failed_at"] = utc_now()
        manifest["error"] = f"{type(exc).__name__}: {exc}"
        manifest["partial_files"] = [path.name for path in partial_paths.values() if path.exists()]
        write_manifest(output, manifest)
        raise
    finally:
        lock.unlink(missing_ok=True)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, help="New or empty output directory")
    parser.add_argument("--languages", nargs="+", default=list(LANGUAGES),
                        help="Dataset config names, space- or comma-separated; default all 12")
    parser.add_argument("--train-mib", type=mib_value, default=Decimal("128"),
                        help="Total raw training text MiB shared equally among languages")
    parser.add_argument("--eval-mib", type=mib_value, default=Decimal("4"),
                        help="Raw text MiB for EACH of calibration/test, shared among languages")
    parser.add_argument("--max-raw-mib", type=mib_value, default=Decimal("160"),
                        help="Hard total raw text cap, including whole-document overshoot")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--fixture", help="Offline JSONL fixture; no Hugging Face imports or network")
    args = parser.parse_args()
    try:
        manifest = prepare(args)
    except (OSError, ValueError, TypeError, RuntimeError) as exc:
        parser.exit(1, f"Preparation failed: {exc}\n")
    print(json.dumps({"status": manifest["status"], "output": str(Path(args.output).resolve()),
                      "documents": manifest["total_documents"],
                      "raw_text_bytes": manifest["total_raw_text_bytes"]}))


if __name__ == "__main__":
    main()
