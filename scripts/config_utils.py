from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable

import tomllib


def load_toml(path: Path) -> Dict[str, Any]:
    with path.open("rb") as f:
        return tomllib.load(f)


def get_nested(d: Dict[str, Any], keys: Iterable[str], default: Any = None) -> Any:
    cur: Any = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def pick(cli_value: Any, cfg_value: Any, default: Any = None) -> Any:
    """
    Ustal wartosc w kolejnosci:
    1) CLI (jesli nie jest None)
    2) config (jesli nie jest None)
    3) default
    """
    if cli_value is not None:
        return cli_value
    if cfg_value is not None:
        return cfg_value
    return default

