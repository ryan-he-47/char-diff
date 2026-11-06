"""YAML config + dotted CLI overrides (--set a.b=value).

Float overrides must include a decimal point: yaml.safe_load('1e-8') parses as
a STRING, '1.0e-8' as float (simple_point_cloud convention).
"""
from __future__ import annotations

import sys
from typing import Any

import yaml


class Cfg:
    """Dict-backed namespace with attribute access (cfg.model.d_model)."""

    def __init__(self, data: dict[str, Any]):
        for k, v in data.items():
            setattr(self, k, Cfg(v) if isinstance(v, dict) else v)

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for k, v in vars(self).items():
            out[k] = v.to_dict() if isinstance(v, Cfg) else v
        return out


def _set_dotted(d: dict[str, Any], dotted: str, value: Any) -> None:
    parts = dotted.split(".")
    for p in parts[:-1]:
        d = d.setdefault(p, {})
    d[parts[-1]] = value


def load(path: str, overrides=()) -> Cfg:
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    for ov in overrides:
        if "=" not in ov:
            raise ValueError(f"override must be a.b=value, got: {ov}")
        k, v = ov.split("=", 1)
        _set_dotted(data, k, yaml.safe_load(v))
    return Cfg(data)


def parse_cli(default_path: str = "configs/default.yaml") -> Cfg:
    """Usage: python script.py [config.yaml] [--set a.b=value ...]"""
    args = sys.argv[1:]
    path = default_path
    overrides = []
    i = 0
    while i < len(args):
        a = args[i]
        if a.endswith((".yaml", ".yml")):
            path = a
        elif a == "--set":
            i += 1
            overrides.append(args[i])
        else:
            raise ValueError(f"unrecognized arg: {a}")
        i += 1
    return load(path, overrides)
