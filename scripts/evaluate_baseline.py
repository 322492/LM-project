#!/usr/bin/env python3
"""
Ewaluacja jakości tłumaczenia EN->PL dla baseline (NLLB-200):
- referencje: data/splits_random/test.pl
- hipotezy:   outputs/baseline/test.hyp.pl

Metryki:
- BLEU (sacrebleu) - obowiązkowo
- chrF (sacrebleu) - opcjonalnie

Wyniki:
- stdout
- outputs/baseline/metrics.txt
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional, Tuple

from config_utils import get_nested, load_toml, pick


def read_lines(path: Path) -> List[str]:
    with path.open("r", encoding="utf-8", errors="replace") as f:
        return [line.rstrip("\n") for line in f]


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def compute_metrics(hyps: List[str], refs: List[str]) -> Tuple[str, Optional[str]]:
    """
    Zwraca:
    - BLEU string (obowiązkowo)
    - chrF string (opcjonalnie; None jeśli nie uda się policzyć)
    """
    from sacrebleu.metrics import BLEU

    bleu = BLEU()
    bleu_score = bleu.corpus_score(hyps, [refs])
    bleu_line = str(bleu_score)

    chrf_line: Optional[str] = None
    try:
        from sacrebleu.metrics import CHRF

        chrf = CHRF()
        chrf_score = chrf.corpus_score(hyps, [refs])
        chrf_line = str(chrf_score)
    except Exception:
        chrf_line = None

    return bleu_line, chrf_line


def main() -> int:
    ap = argparse.ArgumentParser(description="Ewaluacja baseline EN->PL: BLEU (+ opcjonalnie chrF) przez sacrebleu.")
    ap.add_argument("--config", type=Path, default=Path("configs/default.toml"), help="Plik config TOML.")
    ap.add_argument("--refs", type=Path, default=None, help="Plik referencji PL. (nadpisuje config)")
    ap.add_argument("--hyps", type=Path, default=None, help="Plik hipotez PL. (nadpisuje config)")
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("outputs/baseline/metrics.txt"),
        help="Plik wyjsciowy z metrykami. (domyslnie: outputs/baseline/metrics.txt; nadpisuje config)",
    )
    args = ap.parse_args()

    cfg = load_toml(Path(args.config))

    refs_path = Path(pick(args.refs, get_nested(cfg, ["paths", "splits_random_test_pl"]), "data/splits_random/test.pl"))
    hyps_path = Path(pick(args.hyps, get_nested(cfg, ["paths", "baseline_output_pl"]), "outputs/baseline/test.hyp.pl"))
    out_path = Path(pick(args.out, get_nested(cfg, ["paths", "baseline_metrics_txt"]), "outputs/baseline/metrics.txt"))

    refs = read_lines(refs_path)
    hyps = read_lines(hyps_path)

    n_refs = len(refs)
    n_hyps = len(hyps)
    n = min(n_refs, n_hyps)

    if n_refs != n_hyps:
        print("WARNING: rozna liczba linii w refs i hyps.")
        print(f"- refs: {n_refs}")
        print(f"- hyps: {n_hyps}")
        print(f"Metryki policze na min(refs, hyps) = {n} liniach (po indeksach 0..{n-1}).")
        print()

    refs_eval = refs[:n]
    hyps_eval = hyps[:n]

    try:
        bleu_line, chrf_line = compute_metrics(hyps_eval, refs_eval)
    except Exception as e:
        print("ERROR: nie udalo sie policzyc metryk. Czy masz zainstalowane sacrebleu?")
        print(f"Szczegoly: {type(e).__name__}: {e}")
        return 2

    lines = []
    lines.append("=== BASELINE METRICS (EN->PL) ===")
    lines.append(f"refs: {refs_path}")
    lines.append(f"hyps: {hyps_path}")
    lines.append(f"lines_used: {n} (refs={n_refs}, hyps={n_hyps})")
    lines.append("")
    lines.append(f"BLEU: {bleu_line}")
    if chrf_line is not None:
        lines.append(f"chrF: {chrf_line}")
    else:
        lines.append("chrF: (pominięte / niedostępne)")

    report = "\n".join(lines) + "\n"
    print(report)
    write_text(out_path, report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

