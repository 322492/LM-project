#!/usr/bin/env python3
"""
Sprawdzenie duplikatow w korpusie rownoleglym EN-PL w formacie Moses (1 linia = 1 segment).

Skrypt liczy:
- duplikaty identycznych linii po stronie EN,
- duplikaty identycznych linii po stronie PL,
- duplikaty identycznych par (EN, PL).

Uwaga: porownania sa "exact match" (bez normalizacji/oczyszczania) poza usunieciem znaku konca linii.
Skrypt niczego nie modyfikuje i niczego nie zapisuje - tylko raportuje.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import DefaultDict, Dict, Iterable, List, Sequence, Tuple


@dataclass(frozen=True)
class DupSummary:
    total: int
    unique: int
    duplicates: int
    dup_pct: float


def _read_lines(path: Path) -> List[str]:
    # Exact match: usuwamy tylko '\n' z konca; reszta zostaje jak w pliku.
    with path.open("r", encoding="utf-8", errors="replace") as f:
        return [line.rstrip("\n") for line in f]


def _dup_summary(items: Sequence[object]) -> DupSummary:
    total = len(items)
    c = Counter(items)
    unique = len(c)
    duplicates = total - unique
    dup_pct = (100.0 * duplicates / total) if total else 0.0
    return DupSummary(total=total, unique=unique, duplicates=duplicates, dup_pct=dup_pct)


def _collect_duplicate_examples(
    items: Sequence[str],
    max_examples: int,
    max_indices_per_example: int,
) -> List[Tuple[str, List[int]]]:
    """
    Zwraca liste (tekst, [indeksy]) tylko dla elementow, ktore wystepuja >= 2 razy.
    Indeksy sa 1-based (zgodne z numerami linii).
    """
    idxs: DefaultDict[str, List[int]] = defaultdict(list)
    for i, s in enumerate(items, start=1):
        # Trzymamy ograniczona liczbe indeksow na przyklad, zeby nie zalewac logow.
        if len(idxs[s]) < max_indices_per_example:
            idxs[s].append(i)

    # Odfiltruj tylko duplikaty i posortuj po liczbie wystapien (malejaco)
    counts = Counter(items)
    dups = [(s, idxs[s]) for s, cnt in counts.items() if cnt >= 2]
    dups.sort(key=lambda t: (-counts[t[0]], t[0]))
    return dups[:max_examples]


def _collect_duplicate_pair_examples(
    en: Sequence[str],
    pl: Sequence[str],
    max_examples: int,
    max_indices_per_example: int,
) -> List[Tuple[Tuple[str, str], List[int]]]:
    pairs = list(zip(en, pl))
    idxs: DefaultDict[Tuple[str, str], List[int]] = defaultdict(list)
    for i, pair in enumerate(pairs, start=1):
        if len(idxs[pair]) < max_indices_per_example:
            idxs[pair].append(i)

    counts = Counter(pairs)
    dups = [(pair, idxs[pair]) for pair, cnt in counts.items() if cnt >= 2]
    dups.sort(key=lambda t: (-counts[t[0]], t[0][0], t[0][1]))
    return dups[:max_examples]


def _short_lines_stats(items: Sequence[str], max_words: int) -> Tuple[int, Dict[str, int]]:
    """
    Liczy linie o dlugosci <= max_words (split po bialych znakach) oraz ich powtorzenia.
    Zwraca (ile takich linii, licznik powtorzen).
    """
    short = []
    for s in items:
        w = len(s.strip().split()) if s.strip() else 0
        if w <= max_words:
            short.append(s)
    return len(short), dict(Counter(short))


def main() -> int:
    ap = argparse.ArgumentParser(description="Sprawdzenie duplikatow w korpusie Moses EN-PL.")
    ap.add_argument("--en", type=Path, default=Path("data/raw/bible-uedin.en-pl.en"), help="Plik EN (.en).")
    ap.add_argument("--pl", type=Path, default=Path("data/raw/bible-uedin.en-pl.pl"), help="Plik PL (.pl).")
    ap.add_argument("--examples", type=int, default=8, help="Ile przykladow duplikatow wypisac (domyslnie: 8).")
    ap.add_argument(
        "--indices-per-example",
        type=int,
        default=6,
        help="Maks. liczba indeksow pokazana dla jednego duplikatu (domyslnie: 6).",
    )
    ap.add_argument(
        "--short-max-words",
        type=int,
        default=2,
        help="Prog 'bardzo krotkie wersety' (<= N slow). Ustaw 0, zeby pominac (domyslnie: 2).",
    )
    args = ap.parse_args()

    en_lines = _read_lines(args.en)
    pl_lines = _read_lines(args.pl)

    if len(en_lines) != len(pl_lines):
        print(f"BLAD: rozna liczba linii: EN={len(en_lines)}, PL={len(pl_lines)}")
        return 1

    pairs = list(zip(en_lines, pl_lines))

    print("=== DUPLICATES CHECK: korpus rownolegly EN-PL (Moses) ===")
    print(f"EN: {args.en}")
    print(f"PL: {args.pl}")
    print(f"Liczba par (linii): {len(pairs)}")
    print()

    # Statystyki
    en_sum = _dup_summary(en_lines)
    pl_sum = _dup_summary(pl_lines)
    pair_sum = _dup_summary(pairs)

    print("== Podsumowanie ==")
    print(f"EN:   unique={en_sum.unique} / total={en_sum.total} | duplicates={en_sum.duplicates} ({en_sum.dup_pct:.2f}%)")
    print(f"PL:   unique={pl_sum.unique} / total={pl_sum.total} | duplicates={pl_sum.duplicates} ({pl_sum.dup_pct:.2f}%)")
    print(f"PAIR: unique={pair_sum.unique} / total={pair_sum.total} | duplicates={pair_sum.duplicates} ({pair_sum.dup_pct:.2f}%)")
    print()

    # Przyklady
    k = int(args.examples)
    m = int(args.indices_per_example)

    print("== Przyklady duplikatow (EN) ==")
    for text, idx_list in _collect_duplicate_examples(en_lines, max_examples=k, max_indices_per_example=m):
        print("-" * 60)
        print(f"Indeksy (1-based): {idx_list}")
        print(f"EN: {text}")
    if k > 0:
        print("-" * 60)
    print()

    print("== Przyklady duplikatow (PL) ==")
    for text, idx_list in _collect_duplicate_examples(pl_lines, max_examples=k, max_indices_per_example=m):
        print("-" * 60)
        print(f"Indeksy (1-based): {idx_list}")
        print(f"PL: {text}")
    if k > 0:
        print("-" * 60)
    print()

    print("== Przyklady duplikatow (PARA EN,PL) ==")
    for (en_text, pl_text), idx_list in _collect_duplicate_pair_examples(
        en_lines, pl_lines, max_examples=k, max_indices_per_example=m
    ):
        print("-" * 60)
        print(f"Indeksy (1-based): {idx_list}")
        print(f"EN: {en_text}")
        print(f"PL: {pl_text}")
    if k > 0:
        print("-" * 60)
    print()

    # Opcjonalnie: bardzo krotkie wersety
    short_n = int(args.short_max_words)
    if short_n > 0:
        en_short_count, en_short_counter = _short_lines_stats(en_lines, max_words=short_n)
        pl_short_count, pl_short_counter = _short_lines_stats(pl_lines, max_words=short_n)

        def top_repeats(counter: Dict[str, int], top_k: int = 10) -> List[Tuple[str, int]]:
            return sorted(counter.items(), key=lambda t: (-t[1], t[0]))[:top_k]

        print(f"== Bardzo krotkie linie (<= {short_n} slow) ==")
        print(f"EN: {en_short_count} linii | unikalnych: {len(en_short_counter)}")
        print("Top powtorzenia EN (tekst -> liczba wystapien):")
        for s, cnt in top_repeats(en_short_counter, top_k=10):
            if cnt >= 2:
                print(f"- {cnt}x: {s!r}")
        print()
        print(f"PL: {pl_short_count} linii | unikalnych: {len(pl_short_counter)}")
        print("Top powtorzenia PL (tekst -> liczba wystapien):")
        for s, cnt in top_repeats(pl_short_counter, top_k=10):
            if cnt >= 2:
                print(f"- {cnt}x: {s!r}")
        print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# ---------------------------------------------------------------------------
# Jak interpretowac wyniki?
#
# - Duplikaty identycznych linii (EN lub PL) oznaczaja, ze pewne wersety/segmenty
#   powtarzaja sie doslownie w korpusie. Przy losowym splicie po wersetach moze to
#   prowadzic do "przecieku" (ten sam tekst pojawi sie w train i test).
#
# - Duplikaty identycznych par (EN,PL) sa najsilniejszym sygnalem potencjalnego przecieku,
#   bo dokladnie ta sama para tlumaczeniowa powtarza sie w danych.
#
# - Wysoki procent duplikatow (zwlaszcza par) jest sygnalem, ze losowy split po wersetach
#   moze zawyzac wyniki (identyczne teksty moga trafic do train i test).
