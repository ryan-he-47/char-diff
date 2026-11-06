"""Reproducible UTF-8 AE windows from immutable multilingual story records.

Natural sampling: language (configured weights), document (uniform within the
language), code-point start (uniform), then a feasible code-point end (uniform).
The 32-byte limit is a byte limit, not a character count. Unicode normalization
and tokenization are intentionally absent. Replacement targets are the replaced
strings themselves. JSONL input keeps offsets and a bounded document cache;
sampled windows and their vectors are never materialized as a corpus.
"""
from __future__ import annotations

from array import array
from bisect import bisect_right
from collections import OrderedDict
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import random
from typing import Mapping, Sequence


def character_replacement_probability(byte_probability, byte_width):
    """Probability any original byte triggers replacement of its whole character."""
    if not math.isfinite(byte_probability) or not 0 <= byte_probability <= 1:
        raise ValueError("byte_probability must lie in [0, 1]")
    if not isinstance(byte_width, int) or not 1 <= byte_width <= 4:
        raise ValueError("byte_width must be in [1, 4]")
    return 1. - (1. - byte_probability) ** byte_width


@dataclass(frozen=True)
class WindowSample:
    data: bytes                 # AE input AND reconstruction target
    original: bytes
    language: str
    doc_id: str | None
    start_byte: int | None
    end_byte: int | None
    augmentation_selected: bool = False
    substitutions: int = 0
    rejected_substitutions: int = 0
    practice: bool = False


class UTF8WindowSampler:
    """Sample nonempty valid UTF-8 windows; state includes all sampling RNG.

    ``source`` is a JSONL path or a sequence of dictionaries with the keys
    ``id, language, split, text``. Only ``split`` contributes documents and the
    character inventory. ``replace_probability`` is a probability per original
    BYTE: a b-byte character triggers with probability 1-(1-p)**b, exactly the
    union of independent byte triggers. A triggered character is replaced once,
    uniformly by an *other* code point in the inventory. A replacement which would exceed ``max_bytes``
    is rejected, retaining that original character. Single-character practice
    cycles through each language's observed characters without corruption.
    """

    def __init__(self, source: str | Path | Sequence[Mapping], *, split="train",
                 max_bytes=32, seed=1234, language_weights=None,
                 augment_probability=.5, replace_probability=.05,
                 single_character_probability=.01, cache_documents=64,
                 preload_texts=False):
        if not isinstance(max_bytes, int) or not 1 <= max_bytes <= 32:
            raise ValueError("max_bytes must be an integer in [1, 32]")
        for name, value in (("augment_probability", augment_probability),
                            ("replace_probability", replace_probability),
                            ("single_character_probability", single_character_probability)):
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{name} must lie in [0, 1]")
        if not isinstance(cache_documents, int) or cache_documents < 1:
            raise ValueError("cache_documents must be positive")
        if not isinstance(preload_texts, bool):
            raise ValueError("preload_texts must be boolean")
        self.max_bytes = max_bytes
        self.split = str(split)
        self.rng = random.Random(seed)
        self.augment_probability = float(augment_probability)
        self.replace_probability = float(replace_probability)
        self.single_character_probability = float(single_character_probability)
        self.cache_documents = cache_documents
        self.preload_texts = preload_texts
        self.path = Path(source) if isinstance(source, (str, Path)) else None
        self._file = None
        self._documents = []
        self._by_language = {}
        self._cache = OrderedDict()
        self._memory = []
        characters = {}
        fingerprint = hashlib.sha256()

        def accept(record, offset):
            if record.get("split") != self.split:
                return
            if not all(k in record for k in ("id", "language", "text")):
                raise ValueError("Selected records require id, language, split and text")
            text = record["text"]
            language = record["language"]
            if not isinstance(text, str) or not isinstance(language, str) or not language:
                raise ValueError("text and nonempty language must be strings")
            text.encode("utf-8", errors="strict")  # Reject surrogate-containing inputs.
            eligible = set(text) if max_bytes >= 4 else {c for c in text if len(c.encode("utf-8")) <= max_bytes}
            if not eligible:
                return
            doc_id = str(record["id"])
            canonical = {"id": doc_id, "language": language,
                         "split": self.split, "text": text}
            fingerprint.update(json.dumps(canonical, ensure_ascii=False,
                               sort_keys=True, separators=(",", ":")).encode("utf-8") + b"\n")
            index = len(self._documents)
            self._documents.append((doc_id, language, offset))
            if self.path is None or self.preload_texts:
                self._memory.append(text)
            self._by_language.setdefault(language, []).append(index)
            characters.setdefault(language, set()).update(eligible)

        if self.path is not None:
            with self.path.open("rb") as handle:
                while True:
                    offset = handle.tell()
                    line = handle.readline()
                    if not line:
                        break
                    if line.strip():
                        accept(json.loads(line.decode("utf-8")), offset)
        else:
            for record in source:
                accept(record, None)
        if not self._documents:
            raise ValueError(f"No usable documents for split {self.split!r}")
        self.source_sha256 = fingerprint.hexdigest()
        available = sorted(self._by_language)
        if language_weights is None:
            language_weights = {lang: 1. for lang in available}
        if not language_weights or set(language_weights) - set(available):
            raise ValueError("Language weights must name available languages")
        if any(not math.isfinite(v) or v < 0 for v in language_weights.values()):
            raise ValueError("Language weights must be finite and nonnegative")
        self.languages = tuple(lang for lang in available if language_weights.get(lang, 0) > 0)
        if not self.languages:
            raise ValueError("At least one language needs positive weight")
        total = sum(language_weights[lang] for lang in self.languages)
        self.weights = tuple(float(language_weights[lang]) / total for lang in self.languages)
        self._language_characters = {lang: tuple(sorted(characters[lang])) for lang in self.languages}
        self.characters = tuple(sorted(set().union(*(characters[lang] for lang in self.languages))))
        self._character_indices = {char: i for i, char in enumerate(self.characters)}
        self._character_bytes = {char: len(char.encode("utf-8")) for char in self.characters}
        self._character_cursors = {lang: 0 for lang in self.languages}
        self.stats = {"windows": 0, "source_bytes": 0, "target_bytes": 0,
                      "augmentation_selected": 0, "substitutions": 0,
                      "rejected_substitutions": 0, "practice_windows": 0,
                      "languages": {lang: {"windows": 0, "target_bytes": 0}
                                    for lang in self.languages},
                      "replacement_by_width": {
                          str(width): {"characters": 0, "source_bytes": 0,
                                       "triggered": 0, "accepted": 0,
                                       "rejected_length": 0, "no_alternative": 0}
                          for width in range(1, 5)}}

    @property
    def protocol(self):
        return {"version": 2, "source_sha256": self.source_sha256, "split": self.split,
                "max_bytes": self.max_bytes, "language_weights": dict(zip(self.languages, self.weights)),
                "augment_probability": self.augment_probability,
                "replace_probability": self.replace_probability,
                "replace_probability_basis": "per_original_byte_union_per_codepoint",
                "single_character_probability": self.single_character_probability,
                "sampling": "language_weighted/document_uniform/codepoint_start_uniform/feasible_end_uniform",
                "replacement": "uniform_other_training_codepoint/reject_over_byte_limit"}

    def _document(self, index):
        if index in self._cache:
            self._cache.move_to_end(index)
            return self._cache[index]
        doc_id, language, offset = self._documents[index]
        if self.path is None or self.preload_texts:
            text = self._memory[index]
        else:
            if self._file is None:
                self._file = self.path.open("rb")
            self._file.seek(offset)
            record = json.loads(self._file.readline().decode("utf-8"))
            if str(record["id"]) != doc_id or record["language"] != language or record["split"] != self.split:
                raise ValueError("Indexed source changed; rebuild the sampler")
            text = record["text"]
        # Compact immutable-by-convention indices keep a whole small corpus in
        # RAM. Pick 64-bit only when a document could exceed 32-bit byte offsets.
        offsets = array("I" if len(text) * 4 <= 0xffffffff else "Q", [0])
        for char in text:
            offsets.append(offsets[-1] + len(char.encode("utf-8")))
        starts = range(len(text)) if self.max_bytes >= 4 else tuple(
            i for i in range(len(text)) if offsets[i + 1] - offsets[i] <= self.max_bytes)
        value = (text, offsets, starts)
        self._cache[index] = value
        if len(self._cache) > self.cache_documents:
            self._cache.popitem(last=False)
        return value

    def _replace(self, original):
        chars = list(original)
        size = len(original.encode("utf-8"))
        substitutions = rejected = 0
        for i, char in enumerate(chars):
            width = self._character_bytes[char]
            counts = self.stats["replacement_by_width"][str(width)]
            counts["characters"] += 1
            counts["source_bytes"] += width
            probability = character_replacement_probability(self.replace_probability, width)
            if self.rng.random() >= probability:
                continue
            counts["triggered"] += 1
            if len(self.characters) < 2:
                counts["no_alternative"] += 1
                continue
            old_index = self._character_indices[char]
            pick = self.rng.randrange(len(self.characters) - 1)
            replacement = self.characters[pick + (pick >= old_index)]
            new_size = size - self._character_bytes[char] + self._character_bytes[replacement]
            if new_size > self.max_bytes:
                rejected += 1
                counts["rejected_length"] += 1
                continue
            chars[i] = replacement
            size = new_size
            substitutions += 1
            counts["accepted"] += 1
        return "".join(chars), substitutions, rejected

    def sample(self):
        language = self.rng.choices(self.languages, weights=self.weights, k=1)[0]
        if self.rng.random() < self.single_character_probability:
            inventory = self._language_characters[language]
            cursor = self._character_cursors[language]
            data = inventory[cursor % len(inventory)].encode("utf-8")
            self._character_cursors[language] = cursor + 1
            sample = WindowSample(data, data, language, None, None, None, practice=True)
        else:
            index = self.rng.choice(self._by_language[language])
            doc_id = self._documents[index][0]
            text, offsets, starts = self._document(index)
            start = self.rng.choice(starts)
            last = bisect_right(offsets, offsets[start] + self.max_bytes) - 1
            end = self.rng.randint(start + 1, last)
            original = text[start:end]
            selected = self.rng.random() < self.augment_probability
            changed, substitutions, rejected = self._replace(original) if selected else (original, 0, 0)
            sample = WindowSample(changed.encode("utf-8"), original.encode("utf-8"), language,
                                  doc_id, offsets[start], offsets[end], selected, substitutions, rejected)
        return self._record_sample(sample)

    def _record_sample(self, sample):
        language = sample.language
        self.stats["windows"] += 1
        self.stats["source_bytes"] += len(sample.original)
        self.stats["target_bytes"] += len(sample.data)
        self.stats["augmentation_selected"] += int(sample.augmentation_selected)
        self.stats["substitutions"] += sample.substitutions
        self.stats["rejected_substitutions"] += sample.rejected_substitutions
        self.stats["practice_windows"] += int(sample.practice)
        language_stats = self.stats["languages"][language]
        language_stats["windows"] += 1
        language_stats["target_bytes"] += len(sample.data)
        return sample

    def sample_batch(self, count):
        if not isinstance(count, int) or count < 0:
            raise ValueError("count must be a nonnegative integer")
        return [self.sample() for _ in range(count)]

    def state_dict(self):
        return {"protocol": self.protocol, "random_state": self.rng.getstate(),
                "character_cursors": dict(self._character_cursors), "stats": deepcopy(self.stats)}

    def load_state_dict(self, state):
        if state["protocol"] != self.protocol:
            raise ValueError("Sampler data or sampling protocol differs from checkpoint")
        def tuple_tree(value):
            return tuple(tuple_tree(x) for x in value) if isinstance(value, (tuple, list)) else value
        cursors = state["character_cursors"]
        if set(cursors) != set(self.languages) or any(not isinstance(v, int) or v < 0 for v in cursors.values()):
            raise ValueError("Invalid character cursors")
        self.rng.setstate(tuple_tree(state["random_state"]))
        self._character_cursors = dict(cursors)
        self.stats = deepcopy(state["stats"])

    def close(self):
        if self._file is not None:
            self._file.close()
            self._file = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

