"""Per-project budget plus authoritative GPFS user quota admission."""
from __future__ import annotations
import os
import shutil
import subprocess
from pathlib import Path

GIB = 1024 ** 3

def directory_bytes(path):
    total = 0
    for root, dirs, files in os.walk(path):
        dirs[:] = [d for d in dirs if not (Path(root)/d).is_symlink()]
        for name in files:
            p = Path(root) / name
            if not p.is_symlink():
                total += p.stat().st_size
    return total

def gpfs_quota(path):
    """Read KiB values from GPFS, using the filesystem containing this path."""
    executable = shutil.which("mmlsquota") or "/usr/lpp/mmfs/bin/mmlsquota"
    if not Path(executable).is_file():
        return None
    fs = subprocess.check_output(["df", "-P", str(path)], text=True, timeout=15).splitlines()[-1].split()[0]
    proc = subprocess.run([executable, "-u", os.environ.get("USER", ""), "--block-size", "K", fs],
                          text=True, capture_output=True, timeout=20, check=True)
    for row in proc.stdout.splitlines():
        p = row.split()
        if len(p) >= 5 and p[0] == fs and p[1] == "USR":
            used, soft, hard = [int(v.replace(",", "")) * 1024 for v in p[2:5]]
            ceiling = soft if soft > 0 else hard
            return {"filesystem": fs, "used": used, "soft": soft, "hard": hard,
                    "available": max(0, ceiling-used)}
    raise RuntimeError("GPFS quota response could not be parsed; deployment admission remains closed")

class StorageGuard:
    def __init__(self, root, limit_gib=6.0, reserve_gib=1.0, require_quota=False):
        self.root = Path(root).resolve()
        self.limit = int(limit_gib * GIB)
        self.reserve = int(reserve_gib * GIB)
        self.require_quota = require_quota

    def check(self, additional=0):
        used = directory_bytes(self.root)
        if used + additional > self.limit:
            raise RuntimeError(f"Project storage budget: {(used+additional)/GIB:.2f} GiB requested, {self.limit/GIB:.2f} GiB allowed")
        if shutil.disk_usage(self.root).free < additional + self.reserve:
            raise RuntimeError("Filesystem reserve would be consumed")
        quota = gpfs_quota(self.root)
        if self.require_quota and quota is None:
            raise RuntimeError("An authoritative user quota is required on this deployment")
        if quota is not None and quota["available"] < additional + self.reserve:
            raise RuntimeError(f"User quota admission: {quota['available']/GIB:.2f} GiB available; need {(additional+self.reserve)/GIB:.2f} GiB")
        return {"project_bytes": used, "project_limit_bytes": self.limit, "quota": quota}
