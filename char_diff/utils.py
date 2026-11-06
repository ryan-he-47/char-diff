from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import yaml

ROOT = Path(os.environ.get("CHAR_DIFF_ROOT",Path(__file__).resolve().parents[1])).resolve()

def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()

def load_config(path, overrides=()):
    with open(path, encoding="utf-8-sig") as f:
        # JSON numbers such as 1e-08 are valid floats; PyYAML's YAML 1.1
        # resolver treats that spelling as a string. Use the file's parser.
        cfg = json.load(f) if Path(path).suffix.lower() == ".json" else yaml.safe_load(f)
    for item in overrides:
        key, value = item.split("=", 1)
        d = cfg
        parts = key.split(".")
        for part in parts[:-1]:
            if part not in d or not isinstance(d[part], dict):
                raise KeyError(f"Unknown configuration option: {key}")
            d = d[part]
        if parts[-1] not in d:
            raise KeyError(f"Unknown configuration option: {key}")
        d[parts[-1]] = yaml.safe_load(value)
    return cfg

def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".pending")
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        json.dump(value, f, ensure_ascii=False, indent=2)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
