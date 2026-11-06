"""Bounded dataset probes release their owned Parquet shard on early close."""
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pyarrow as pa
import pyarrow.parquet as pq
from char_diff.cache import raw_rows
from char_diff.utils import ROOT

with tempfile.TemporaryDirectory(prefix="test_download_", dir=ROOT / "runs") as folder:
    root=Path(folder).resolve();root.relative_to(ROOT.resolve())
    path=root/"owned-shard.parquet"
    pq.write_table(pa.table({"text":["first story","second story"]}),path)
    with patch("char_diff.cache.download",return_value=path):
        rows=raw_rows("train",root,None)
        assert next(rows)=="first story"
        rows.close()
    assert not path.exists()
print("PASS early-close probe releases its temporary Parquet shard",flush=True)
