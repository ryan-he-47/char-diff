"""Targeted checks for UTF-8 window sampling and resumability."""
from array import array
from pathlib import Path
import json
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from char_diff.ae_windows import UTF8WindowSampler, character_replacement_probability


RECORDS = [
    {"id": "zh1", "language": "zh", "split": "train", "text": "中文故事🙂第一段。"},
    {"id": "zh2", "language": "zh", "split": "train", "text": "第二个故事🐈，含空格 及零\x00。"},
    {"id": "en1", "language": "en", "split": "train", "text": "A cat sat.\nAnother tale!"},
    {"id": "excluded", "language": "zh", "split": "test", "text": "🛑"},
]


def test_boundaries_and_corruption():
    sampler = UTF8WindowSampler(RECORDS, max_bytes=7, seed=4,
                               augment_probability=1, replace_probability=1,
                               single_character_probability=0)
    docs = {r["id"]: r["text"].encode("utf-8") for r in RECORDS}
    changed = rejected = 0
    for row in sampler.sample_batch(1000):
        source = docs[row.doc_id]
        assert row.original == source[row.start_byte:row.end_byte]
        source[:row.start_byte].decode("utf-8")
        source[row.end_byte:].decode("utf-8")
        assert 1 <= len(row.data) <= 7
        assert row.data.decode("utf-8").encode("utf-8") == row.data
        assert len(row.original.decode("utf-8")) == len(row.data.decode("utf-8"))
        changed += row.substitutions
        rejected += row.rejected_substitutions
    assert changed > 0 and rejected > 0
    assert "🛑" not in sampler.characters
    print("PASS: UTF-8 boundaries, bounded faithful character replacement")


def test_resume():
    sampler = UTF8WindowSampler(RECORDS, seed=10, single_character_probability=.25)
    sampler.sample_batch(73)
    state = json.loads(json.dumps(sampler.state_dict()))
    expected = sampler.sample_batch(150)
    resumed = UTF8WindowSampler(RECORDS, seed=999, single_character_probability=.25)
    resumed.load_state_dict(state)
    assert resumed.sample_batch(150) == expected
    assert resumed.stats == sampler.stats
    mismatched = UTF8WindowSampler(RECORDS, max_bytes=31, single_character_probability=.25)
    try:
        mismatched.load_state_dict(state)
    except ValueError:
        pass
    else:
        raise AssertionError("changed protocol accepted")
    print("PASS: RNG, character cursors and counts resume exactly")


def test_jsonl_and_language_isolation():
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "stories.jsonl"
        path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in RECORDS), encoding="utf-8")
        kwargs = dict(seed=3, language_weights={"zh": 1, "en": 0},
                      augment_probability=0, single_character_probability=0, cache_documents=1)
        memory = UTF8WindowSampler(RECORDS, **kwargs)
        with UTF8WindowSampler(path, **kwargs) as indexed:
            rows = indexed.sample_batch(300)
            assert rows == memory.sample_batch(300)
            assert all(row.language == "zh" and row.doc_id in {"zh1", "zh2"} for row in rows)
            assert all(row.data == row.original and len(row.data) <= 32 for row in rows)
            assert len(indexed._cache) == 1
            assert indexed.source_sha256 == memory.source_sha256
        practice = UTF8WindowSampler(path, language_weights={"en": 1}, single_character_probability=1)
        inventory = sorted(set(RECORDS[2]["text"]))
        assert [r.data.decode("utf-8") for r in practice.sample_batch(len(inventory))] == inventory
        practice.close()
    print("PASS: indexed JSONL, language/split isolation, per-document boundaries")


def test_byte_balanced_replacement():
    assert abs(character_replacement_probability(.05, 1) - .05) < 1e-12
    assert abs(character_replacement_probability(.05, 3) - .142625) < 1e-12
    assert character_replacement_probability(0, 4) == 0
    assert character_replacement_probability(1, 4) == 1
    # The same U=.10 fails a 1-byte trigger and passes a 3-byte trigger.
    class FixedRandom:
        def random(self): return .10
        def randrange(self, stop): return 0
    sampler = UTF8WindowSampler(RECORDS, replace_probability=.05)
    sampler.rng = FixedRandom()
    output, accepted, rejected = sampler._replace("A中")
    assert output[0] == "A" and output[1] != "中" and accepted == 1 and rejected == 0
    buckets = sampler.stats["replacement_by_width"]
    assert buckets["1"]["triggered"] == 0
    assert buckets["3"]["triggered"] == buckets["3"]["accepted"] == 1
    assert buckets["3"]["source_bytes"] == 3
    print("PASS: byte-balanced character triggers and width-stratified counts")


def test_preload_and_compact_indices():
    """Storage/cache changes preserve every sample and resumable RNG field."""
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "stories.jsonl"
        path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in RECORDS), encoding="utf-8")
        kwargs = dict(seed=73, single_character_probability=.1,
                      augment_probability=.8, replace_probability=.3)
        with UTF8WindowSampler(path, preload_texts=False, cache_documents=1, **kwargs) as disk, \
                UTF8WindowSampler(path, preload_texts=True, cache_documents=131072, **kwargs) as memory:
            assert disk.protocol == memory.protocol
            assert disk.sample_batch(600) == memory.sample_batch(600)
            assert disk.state_dict() == memory.state_dict()
            assert memory._file is None and len(memory._memory) == 3
            for _, offsets, starts in memory._cache.values():
                assert isinstance(offsets, array) and offsets.typecode == "I"
                assert isinstance(starts, range)
            disk_state, memory_state = disk.state_dict(), memory.state_dict()
            expected = disk.sample_batch(200)
            memory.load_state_dict(disk_state)
            assert memory.sample_batch(200) == expected
            disk.load_state_dict(memory_state)
            assert disk.sample_batch(200) == expected
            assert disk.state_dict() == memory.state_dict()
        # The filtered-start path also retains its original sampling contract.
        with UTF8WindowSampler(path, max_bytes=2, cache_documents=1, **kwargs) as disk, \
                UTF8WindowSampler(path, max_bytes=2, preload_texts=True, cache_documents=131072, **kwargs) as memory:
            assert disk.sample_batch(100) == memory.sample_batch(100)
            assert disk.state_dict() == memory.state_dict()
    print("PASS: preload/cache/compact-index sample identity and cross-mode exact resume")


if __name__ == "__main__":
    if "--preload-only" in sys.argv:
        test_preload_and_compact_indices()
    else:
        test_boundaries_and_corruption()
        test_resume()
        test_jsonl_and_language_isolation()
        test_byte_balanced_replacement()
        test_preload_and_compact_indices()
