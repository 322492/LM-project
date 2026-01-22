#!/usr/bin/env python3
"""
Sanity check dla korpusu równoległego EN–PL w formacie Moses (1 linia = 1 segment).

Skrypt:
- wczytuje parę plików EN i PL (domyślnie auto-wykrywa w data/raw/),
- sprawdza spójność (liczba linii, puste segmenty, brakujące linie),
- liczy statystyki długości (w słowach: split po białych znakach) i percentyle,
- losuje kilka par EN–PL do ręcznej kontroli,
- NIE modyfikuje danych (tylko odczyt i raport).
"""

from __future__ import annotations

import argparse
import itertools
import random
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median
from typing import Iterable, List, Optional, Sequence, Tuple


@dataclass
class LengthStats:
    n: int
    min_: int
    max_: int
    avg: float
    med: float
    percentiles: List[Tuple[int, float]]  # (p, value)


def _word_len(line: str) -> int:
    # Moses: liczymy długość jako liczbę "słów" po split() na białych znakach.
    s = line.strip()
    if not s:
        return 0
    return len(s.split())


def _compute_percentiles(values: Sequence[int], ps: Sequence[int]) -> List[Tuple[int, float]]:
    """
    Prosty percentyl: nearest-rank na posortowanej liście.
    Zwraca float, żeby czytelnie wypisać wyniki nawet przy uśrednianiu w przyszłości.
    """
    if not values:
        return [(p, float("nan")) for p in ps]

    xs = sorted(values)
    n = len(xs)
    out: List[Tuple[int, float]] = []
    for p in ps:
        if p <= 0:
            out.append((p, float(xs[0])))
            continue
        if p >= 100:
            out.append((p, float(xs[-1])))
            continue
        # nearest-rank: k = ceil(p/100 * n), indeks 1..n
        k = int((p / 100) * n)
        if (p / 100) * n > k:
            k += 1
        k = max(1, min(n, k))
        out.append((p, float(xs[k - 1])))
    return out


def _summarize_lengths(lengths: Sequence[int], percentiles: Sequence[int]) -> LengthStats:
    if not lengths:
        return LengthStats(
            n=0,
            min_=0,
            max_=0,
            avg=float("nan"),
            med=float("nan"),
            percentiles=_compute_percentiles([], percentiles),
        )

    return LengthStats(
        n=len(lengths),
        min_=min(lengths),
        max_=max(lengths),
        avg=float(mean(lengths)),
        med=float(median(lengths)),
        percentiles=_compute_percentiles(lengths, percentiles),
    )


def _auto_detect_pair(raw_dir: Path) -> Tuple[Path, Path]:
    en_files = sorted(raw_dir.glob("*.en"))
    pl_files = sorted(raw_dir.glob("*.pl"))

    if len(en_files) == 1 and len(pl_files) == 1:
        return en_files[0], pl_files[0]

    # Spróbuj dopasować po wspólnym prefiksie (np. *.en i *.pl z tą samą nazwą bazową)
    pairs: List[Tuple[Path, Path]] = []
    pl_by_stem = {p.with_suffix("").name: p for p in pl_files}
    for en in en_files:
        stem = en.with_suffix("").name
        if stem in pl_by_stem:
            pairs.append((en, pl_by_stem[stem]))

    if len(pairs) == 1:
        return pairs[0]

    raise FileNotFoundError(
        "Nie udało się jednoznacznie wykryć pary plików .en/.pl w data/raw/. "
        "Podaj --en-file i --pl-file ręcznie."
    )


def _reservoir_update(
    reservoir: List[Tuple[int, str, str]],
    item: Tuple[int, str, str],
    seen: int,
    k: int,
    rng: random.Random,
) -> None:
    """
    Reservoir sampling: po przetworzeniu N elementów, każdy ma jednakową szansę znaleźć się w próbce.
    """
    if k <= 0:
        return
    if len(reservoir) < k:
        reservoir.append(item)
        return
    j = rng.randint(1, seen)  # 1..seen
    if j <= k:
        reservoir[j - 1] = item


def run_sanity_check(
    en_path: Path,
    pl_path: Path,
    sample_k: int,
    seed: int,
    short_words: int,
    long_words: int,
    percentiles: Sequence[int],
) -> int:
    en_lengths: List[int] = []
    pl_lengths: List[int] = []

    empty_en = 0
    empty_pl = 0
    empty_any = 0
    missing_en = 0  # linia jest w PL, a brak w EN
    missing_pl = 0  # linia jest w EN, a brak w PL

    short_en = 0
    short_pl = 0
    long_en = 0
    long_pl = 0

    rng = random.Random(seed)
    sample: List[Tuple[int, str, str]] = []  # (line_no, en, pl)

    # Uwaga: zip_longest pozwala wykryć różną liczbę linii.
    with en_path.open("r", encoding="utf-8", errors="replace") as f_en, pl_path.open(
        "r", encoding="utf-8", errors="replace"
    ) as f_pl:
        for i, (en_line, pl_line) in enumerate(itertools.zip_longest(f_en, f_pl), start=1):
            if en_line is None:
                missing_en += 1
                en_line = ""
            if pl_line is None:
                missing_pl += 1
                pl_line = ""

            en_s = en_line.rstrip("\n")
            pl_s = pl_line.rstrip("\n")

            en_is_empty = len(en_s.strip()) == 0
            pl_is_empty = len(pl_s.strip()) == 0

            if en_is_empty:
                empty_en += 1
            if pl_is_empty:
                empty_pl += 1
            if en_is_empty or pl_is_empty:
                empty_any += 1

            en_w = _word_len(en_s)
            pl_w = _word_len(pl_s)
            en_lengths.append(en_w)
            pl_lengths.append(pl_w)

            if en_w <= short_words:
                short_en += 1
            if pl_w <= short_words:
                short_pl += 1
            if en_w >= long_words:
                long_en += 1
            if pl_w >= long_words:
                long_pl += 1

            _reservoir_update(sample, (i, en_s, pl_s), seen=i, k=sample_k, rng=rng)

    n_pairs = max(len(en_lengths), len(pl_lengths))
    en_stats = _summarize_lengths(en_lengths, percentiles)
    pl_stats = _summarize_lengths(pl_lengths, percentiles)

    def fmt_ps(ps: List[Tuple[int, float]]) -> str:
        return ", ".join([f"p{p}={v:.0f}" if v == v else f"p{p}=nan" for p, v in ps])

    print("=== SANITY CHECK: korpus równoległy EN–PL (Moses) ===")
    print(f"EN file: {en_path}")
    print(f"PL file: {pl_path}")
    print()

    print("== Spójność ==")
    print(f"Liczba linii (przetworzonych par): {n_pairs}")
    if missing_en or missing_pl:
        print(f"UWAGA: różna liczba linii w plikach:")
        print(f"- brakujących linii w EN (PL ma więcej): {missing_en}")
        print(f"- brakujących linii w PL (EN ma więcej): {missing_pl}")
    else:
        print("OK: oba pliki mają tę samą liczbę linii.")

    print()
    print("== Puste segmenty ==")
    print(f"Puste linie EN: {empty_en}")
    print(f"Puste linie PL: {empty_pl}")
    print(f"Puste w dowolnej stronie (EN lub PL): {empty_any}")

    print()
    print("== Długości (w słowach) ==")
    print(f"Progi (konfigurowalne): short <= {short_words}, long >= {long_words}")
    print(f"Krótkie EN: {short_en} | Długie EN: {long_en}")
    print(f"Krótkie PL: {short_pl} | Długie PL: {long_pl}")
    print()

    print("EN stats:")
    print(f"- n={en_stats.n}, min={en_stats.min_}, avg={en_stats.avg:.2f}, med={en_stats.med:.2f}, max={en_stats.max_}")
    print(f"- percentyle: {fmt_ps(en_stats.percentiles)}")
    print()

    print("PL stats:")
    print(f"- n={pl_stats.n}, min={pl_stats.min_}, avg={pl_stats.avg:.2f}, med={pl_stats.med:.2f}, max={pl_stats.max_}")
    print(f"- percentyle: {fmt_ps(pl_stats.percentiles)}")
    print()

    print(f"== Losowe przykłady (k={sample_k}, seed={seed}) ==")
    for line_no, en_s, pl_s in sorted(sample, key=lambda x: x[0]):
        print("-" * 60)
        print(f"[{line_no}] EN: {en_s}")
        print(f"[{line_no}] PL: {pl_s}")
    if sample_k > 0:
        print("-" * 60)

    # Kod zwrotu: 0 = OK, 1 = zauważone problemy spójności (różna liczba linii)
    return 1 if (missing_en or missing_pl) else 0


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Sanity check korpusu równoległego EN–PL (Moses: 1 linia = 1 segment).",
    )
    p.add_argument(
        "--raw-dir",
        type=Path,
        default=Path("data/raw"),
        help="Katalog z plikami źródłowymi (domyślnie: data/raw).",
    )
    p.add_argument(
        "--en-file",
        type=Path,
        default=None,
        help="Ścieżka do pliku EN (.en). Jeśli nie podasz, skrypt spróbuje wykryć plik w --raw-dir.",
    )
    p.add_argument(
        "--pl-file",
        type=Path,
        default=None,
        help="Ścieżka do pliku PL (.pl). Jeśli nie podasz, skrypt spróbuje wykryć plik w --raw-dir.",
    )
    p.add_argument("--samples", type=int, default=8, help="Ile par wypisać do ręcznej kontroli (domyślnie: 8).")
    p.add_argument("--seed", type=int, default=123, help="Ziarno losowania przykładów (domyślnie: 123).")
    p.add_argument(
        "--short-words",
        type=int,
        default=1,
        help="Próg 'bardzo krótka linia' w słowach (<=). Domyślnie: 1.",
    )
    p.add_argument(
        "--long-words",
        type=int,
        default=80,
        help="Próg 'bardzo długa linia' w słowach (>=). Domyślnie: 80.",
    )
    p.add_argument(
        "--percentiles",
        type=str,
        default="1,5,10,25,50,75,90,95,99",
        help="Lista percentyli do policzenia, np. '1,5,50,95,99'.",
    )
    return p


def main() -> int:
    args = build_arg_parser().parse_args()

    raw_dir: Path = args.raw_dir
    en_path: Optional[Path] = args.en_file
    pl_path: Optional[Path] = args.pl_file

    if en_path is None or pl_path is None:
        en_auto, pl_auto = _auto_detect_pair(raw_dir)
        en_path = en_path or en_auto
        pl_path = pl_path or pl_auto

    en_path = en_path.resolve()
    pl_path = pl_path.resolve()

    percentiles = [int(x.strip()) for x in str(args.percentiles).split(",") if x.strip()]
    percentiles = sorted(set(percentiles))

    return run_sanity_check(
        en_path=en_path,
        pl_path=pl_path,
        sample_k=int(args.samples),
        seed=int(args.seed),
        short_words=int(args.short_words),
        long_words=int(args.long_words),
        percentiles=percentiles,
    )


if __name__ == "__main__":
    raise SystemExit(main())

# ---------------------------------------------------------------------------
# Jak interpretować wyniki sanity checku i kiedy dane uznać za problematyczne?
#
# 1) Różna liczba linii EN i PL
#    - To zazwyczaj krytyczny problem w formacie Moses, bo pary przestają się zgadzać.
#    - Jeśli to się dzieje: najpierw ustal, czy pliki pochodzą z tego samego źródła
#      i czy nie zostały ręcznie edytowane / ucięte.
#
# 2) Puste segmenty (EN lub PL)
#    - Pojedyncze puste linie mogą się zdarzyć, ale duża liczba pustych par często
#      wskazuje na błędy ekstrakcji/segmentacji lub nieprawidłowy preprocessing.
#
# 3) Bardzo krótkie / bardzo długie linie
#    - Dużo bardzo krótkich linii może oznaczać, że segmentacja jest zbyt agresywna
#      (np. rozbijanie na pojedyncze słowa) lub są artefakty (nagłówki, numeracja).
#    - Dużo bardzo długich linii może oznaczać brak segmentacji (np. całe akapity)
#      albo wklejone listy/ciągi znaków.
#    - Progi short/long są konfigurowalne, więc warto je dostosować do charakteru danych,
#      ale dopiero po obejrzeniu losowych przykładów.
#
# 4) Percentyle długości
#    - Percentyle (np. p95, p99) pomagają szybko zobaczyć „ogon” rozkładu długości.
#    - Jeśli p99 jest bardzo wysoki, a losowe przykłady pokazują nietypowe segmenty,
#      to dane mogą wymagać lepszej segmentacji/oczyszczenia (bez robienia tego tutaj).
