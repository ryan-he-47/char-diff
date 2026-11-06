"""Check project budget and GPFS user quota before creating large artifacts."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from char_diff.storage import StorageGuard, GIB
from char_diff.utils import ROOT

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--additional-gib", type=float, default=0)
    p.add_argument("--limit-gib", type=float, default=16)
    p.add_argument("--reserve-gib", type=float, default=1)
    p.add_argument("--require-quota", action="store_true")
    args = p.parse_args()
    result = StorageGuard(ROOT, args.limit_gib, args.reserve_gib, args.require_quota).check(
        int(args.additional_gib * GIB))
    print(json.dumps(result, indent=2))
