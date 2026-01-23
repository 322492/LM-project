#!/usr/bin/env python3
"""
Losowy (po liniach) split 80/5/15 dla korpusu rownoleglego EN-PL w formacie Moses.

Wazne cechy:
- split jest deterministyczny (seed),
- split jest na poziomie wersetow/linii (bez deduplikacji i bez filtrowania dlugosci),
- skrypt niczego nie zmienia w data/raw/ (tylko czyta) i zapisuje wyniki do data/splits_random/.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import List, Sequence, Tuple

from config_utils import get_nested, load_toml, pick


def read_lines(path: Path) -> List[str]:
    with path.open("r", encoding="utf-8", errors="replace") as f:
        return [line.rstrip("\n") for line in f]


def warn_if_empty_segments(lines: Sequence[str], side: str) -> int:
    empty = sum(1 for s in lines if len(s.strip()) == 0)
    if empty > 0:
        print(f"WARNING: puste segmenty po stronie {side}: {empty}")
    return empty


def split_indices(n: int, seed: int, train_ratio: float, val_ratio: float, test_ratio: float) -> Tuple[List[int], List[int], List[int]]:
    if n <= 0:
        return [], [], []

    ratio_sum = train_ratio + val_ratio + test_ratio
    if abs(ratio_sum - 1.0) > 1e-9:
        raise ValueError(f"Suma proporcji musi wynosic 1.0, a wynosi {ratio_sum}")

    idx = list(range(n))  # 0-based
    rng = random.Random(seed)
    rng.shuffle(idx)

    n_train = int(train_ratio * n)
    n_val = int(val_ratio * n)
    n_test = n - n_train - n_val

    train_idx = idx[:n_train]
    val_idx = idx[n_train : n_train + n_val]
    test_idx = idx[n_train + n_val :]
    assert len(train_idx) + len(val_idx) + len(test_idx) == n
    assert len(test_idx) == n_test

    return train_idx, val_idx, test_idx


def write_split(out_en: Path, out_pl: Path, en_lines: Sequence[str], pl_lines: Sequence[str], indices: Sequence[int]) -> None:
    with out_en.open("w", encoding="utf-8", newline="\n") as f_en, out_pl.open("w", encoding="utf-8", newline="\n") as f_pl:
        for i in indices:
            f_en.write(en_lines[i] + "\n")
            f_pl.write(pl_lines[i] + "\n")


def pct(part: int, total: int) -> float:
    return 100.0 * part / total if total else 0.0


def sample_indices(indices: Sequence[int], k: int, seed: int) -> List[int]:
    if k <= 0 or not indices:
        return []
    rng = random.Random(seed)
    # deterministic: bierzemy probke bez powtorzen (jesli sie da)
    k_eff = min(k, len(indices))
    return sorted(rng.sample(list(indices), k=k_eff))


def main() -> int:
    ap = argparse.ArgumentParser(description="Losowy split 80/5/15 (po liniach) dla Moses EN-PL.")
    ap.add_argument("--config", type=Path, default=Path("configs/default.toml"), help="Plik config TOML.")
    ap.add_argument("--en", type=Path, default=None, help="Plik EN (.en). (nadpisuje config)")
    ap.add_argument("--pl", type=Path, default=None, help="Plik PL (.pl). (nadpisuje config)")
    ap.add_argument("--out-dir", type=Path, default=None, help="Katalog wyjsciowy. (nadpisuje config)")
    ap.add_argument("--seed", type=int, default=None, help="Seed losowosci. (nadpisuje config)")
    ap.add_argument("--train", type=float, default=None, help="Udzial TRAIN. (nadpisuje config)")
    ap.add_argument("--val", type=float, default=None, help="Udzial VAL. (nadpisuje config)")
    ap.add_argument("--test", type=float, default=None, help="Udzial TEST. (nadpisuje config)")
    ap.add_argument("--show-indices", type=int, default=None, help="Ile przykladowych indeksow pokazac na split. (nadpisuje config)")
    args = ap.parse_args()

    cfg = load_toml(Path(args.config))

    en_path = Path(
        pick(args.en, get_nested(cfg, ["paths", "raw_en"]), "data/raw/bible-uedin.en-pl.en")
    )
    pl_path = Path(
        pick(args.pl, get_nested(cfg, ["paths", "raw_pl"]), "data/raw/bible-uedin.en-pl.pl")
    )
    out_dir = Path(
        pick(args.out_dir, get_nested(cfg, ["paths", "splits_random_dir"]), "data/splits_random")
    )
    seed = int(pick(args.seed, get_nested(cfg, ["random_split", "seed"]), 123))
    train_ratio = float(pick(args.train, get_nested(cfg, ["random_split", "train_ratio"]), 0.80))
    val_ratio = float(pick(args.val, get_nested(cfg, ["random_split", "val_ratio"]), 0.05))
    test_ratio = float(pick(args.test, get_nested(cfg, ["random_split", "test_ratio"]), 0.15))
    show_indices = int(pick(args.show_indices, get_nested(cfg, ["random_split", "show_indices"]), 5))

    en_lines = read_lines(en_path)
    pl_lines = read_lines(pl_path)

    if len(en_lines) != len(pl_lines):
        print(f"ERROR: rozna liczba linii EN i PL: EN={len(en_lines)} PL={len(pl_lines)}")
        return 1

    total = len(en_lines)
    warn_if_empty_segments(en_lines, "EN")
    warn_if_empty_segments(pl_lines, "PL")

    train_idx, val_idx, test_idx = split_indices(
        n=total,
        seed=seed,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    write_split(out_dir / "train.en", out_dir / "train.pl", en_lines, pl_lines, train_idx)
    write_split(out_dir / "val.en", out_dir / "val.pl", en_lines, pl_lines, val_idx)
    write_split(out_dir / "test.en", out_dir / "test.pl", en_lines, pl_lines, test_idx)

    print("=== RANDOM SPLIT: Moses EN-PL (line-level) ===")
    print(f"EN: {en_path}")
    print(f"PL: {pl_path}")
    print(f"Output dir: {out_dir}")
    print(f"Seed: {seed}")
    print()
    print(f"TRAIN: {len(train_idx)} ({pct(len(train_idx), total):.2f}%)")
    print(f"VAL:   {len(val_idx)} ({pct(len(val_idx), total):.2f}%)")
    print(f"TEST:  {len(test_idx)} ({pct(len(test_idx), total):.2f}%)")
    print()
    print("Sample indices (0-based):")
    print(f"- train: {sample_indices(train_idx, k=show_indices, seed=seed + 1)}")
    print(f"- val:   {sample_indices(val_idx, k=show_indices, seed=seed + 2)}")
    print(f"- test:  {sample_indices(test_idx, k=show_indices, seed=seed + 3)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

