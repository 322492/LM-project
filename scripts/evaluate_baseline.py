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
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from config_utils import get_nested, load_toml, pick


def read_lines(path: Path) -> List[str]:
    with path.open("r", encoding="utf-8", errors="replace") as f:
        return [line.rstrip("\n") for line in f]


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")

def read_json_if_exists(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None

def read_indices(path: Path) -> List[int]:
    return [int(x.strip()) for x in path.read_text(encoding="utf-8", errors="replace").splitlines() if x.strip()]


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
    hyps_path = Path(pick(args.hyps, get_nested(cfg, ["paths", "baseline_output_pl"]), "outputs/baseline/full_test.hyp.pl"))
    out_path = Path(pick(args.out, get_nested(cfg, ["paths", "baseline_metrics_txt"]), "outputs/baseline/metrics.txt"))

    hyps = read_lines(hyps_path)

    # Sidecar files obok hyps:
    # - *.ref.pl (quick run)
    # - *.indices.txt (sample)
    ref_subset_path = hyps_path.with_name(f"{hyps_path.stem}.ref.pl")
    indices_path = hyps_path.with_name(f"{hyps_path.stem}.indices.txt")
    meta_path = hyps_path.with_name(f"{hyps_path.name}.meta.json")

    indices: Optional[List[int]] = None
    if indices_path.exists():
        indices = read_indices(indices_path)

    # Wybór referencji:
    # 1) jeśli istnieje *.ref.pl i pasuje długością do hyps -> użyj go
    # 2) jeśli istnieje *.indices.txt -> użyj pełnych refs i wybierz te indeksy
    # 3) inaczej wymagamy, żeby refs i hyps miały tę samą liczbę linii
    refs_eval: List[str]
    if ref_subset_path.exists():
        refs_subset = read_lines(ref_subset_path)
        if len(refs_subset) == len(hyps):
            refs_eval = refs_subset
        else:
            refs_eval = []
    else:
        refs_eval = []

    if not refs_eval:
        refs_full = read_lines(refs_path)
        if indices is not None:
            # liczymy dokładnie na indeksach
            refs_eval = [refs_full[i] for i in indices if 0 <= i < len(refs_full)]
            if len(refs_eval) != len(hyps):
                raise SystemExit(
                    f"ERROR: liczba refs po indeksach ({len(refs_eval)}) != liczba hyps ({len(hyps)}). "
                    "Sprawdz plik indices.txt lub ref.pl obok hyps."
                )
        else:
            if len(refs_full) != len(hyps):
                raise SystemExit(
                    f"ERROR: refs ({len(refs_full)}) i hyps ({len(hyps)}) maja rozna liczbe linii, "
                    "a nie znaleziono dopasowanego pliku *.ref.pl obok hyps."
                )
            refs_eval = refs_full

    try:
        bleu_line, chrf_line = compute_metrics(hyps, refs_eval)
    except Exception as e:
        print("ERROR: nie udalo sie policzyc metryk. Czy masz zainstalowane sacrebleu?")
        print(f"Szczegoly: {type(e).__name__}: {e}")
        return 2

    lines = []
    lines.append("=== BASELINE METRICS (EN->PL) ===")
    lines.append(f"refs: {ref_subset_path if ref_subset_path.exists() else refs_path}")
    lines.append(f"hyps: {hyps_path}")
    lines.append(f"lines_used: {len(hyps)}")
    if indices is not None:
        lines.append(f"indices: {indices_path}")
    lines.append("")
    lines.append(f"BLEU: {bleu_line}")
    if chrf_line is not None:
        lines.append(f"chrF: {chrf_line}")
    else:
        lines.append("chrF: (pominięte / niedostępne)")

    report = "\n".join(lines) + "\n"
    print(report)
    write_text(out_path, report)

    # README.txt w outputs/baseline/ (generowany przy ewaluacji)
    meta = read_json_if_exists(meta_path) or {}
    readme_path = Path("outputs/baseline/README.txt")
    readme_lines = []
    readme_lines.append("=== BASELINE RUN SUMMARY ===")
    readme_lines.append(f"date: {datetime.now().isoformat(timespec='seconds')}")
    readme_lines.append(f"hyps: {hyps_path}")
    readme_lines.append(f"refs: {ref_subset_path if ref_subset_path.exists() else refs_path}")
    if indices is not None:
        readme_lines.append(f"indices: {indices_path}")
    readme_lines.append(f"lines_used: {len(hyps)}")
    readme_lines.append("")
    readme_lines.append(f"model: {meta.get('model', get_nested(cfg, ['baseline_nllb', 'model_name']))}")
    readme_lines.append(f"batch_size: {meta.get('batch_size', get_nested(cfg, ['baseline_nllb', 'batch_size']))}")
    readme_lines.append(f"num_beams: {meta.get('num_beams', get_nested(cfg, ['baseline_nllb', 'num_beams']))}")
    readme_lines.append(f"max_new_tokens: {meta.get('max_new_tokens', get_nested(cfg, ['baseline_nllb', 'max_new_tokens']))}")
    readme_lines.append(f"seed: {meta.get('seed', get_nested(cfg, ['baseline_nllb', 'seed']))}")
    readme_lines.append("")
    readme_lines.append(f"BLEU: {bleu_line}")
    if chrf_line is not None:
        readme_lines.append(f"chrF: {chrf_line}")
    else:
        readme_lines.append("chrF: (pominięte / niedostępne)")
    write_text(readme_path, "\n".join(readme_lines) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

