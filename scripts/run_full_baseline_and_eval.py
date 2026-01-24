#!/usr/bin/env python3
"""
UWAGA: CAŁOŚĆ TRWA KILKA GODZIN! - PUSZCZAĆ NA NOC!

Jedna komenda: full baseline inference + ewaluacja metryk.

Uruchamia:
1) scripts/run_baseline_inference.py (full test)
2) scripts/evaluate_baseline.py (BLEU + chrF)

CPU-only, bez trenowania.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Optional, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"


def _run(cmd: list[str]) -> None:
    print(f"[cmd] {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, cwd=str(PROJECT_ROOT), check=True)


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _parse_metrics(metrics_txt: str) -> Tuple[Optional[str], Optional[str]]:
    bleu = None
    chrf = None
    for line in metrics_txt.splitlines():
        if line.startswith("BLEU:"):
            bleu = line[len("BLEU:") :].strip()
        if line.startswith("chrF:"):
            chrf = line[len("chrF:") :].strip()
    return bleu, chrf


def main() -> int:
    # Parametry wymagane przez zadanie:
    output_hyp = Path("outputs/baseline/full_test.hyp.pl")
    output_metrics = Path("outputs/baseline/full_test.metrics.txt")

    # 1) Inference (full)
    _run(
        [
            sys.executable,
            str(SCRIPTS_DIR / "run_baseline_inference.py"),
            "--max-sentences",
            "0",
            "--num-beams",
            "1",
            "--max-new-tokens",
            "96",
            "--batch-size",
            "4",
            "--log-every",
            "256",
            "--output",
            str(output_hyp),
            "--no-save-ref-subset",
        ]
    )

    meta_path = PROJECT_ROOT / f"{output_hyp.name}.meta.json"
    # meta.json jest zapisywane obok outputu (ta sama nazwa + .meta.json)
    meta_path = (PROJECT_ROOT / output_hyp).with_name(f"{output_hyp.name}.meta.json")
    meta = _read_json(meta_path)
    inf_time = float(meta.get("inference_time_s", 0.0))
    lines = int(meta.get("lines_selected", 0))
    speed = (lines / inf_time) if inf_time > 0 else 0.0

    # 2) Ewaluacja
    _run(
        [
            sys.executable,
            str(SCRIPTS_DIR / "evaluate_baseline.py"),
            "--refs",
            "data/splits_random/test.pl",
            "--hyps",
            str(output_hyp),
            "--out",
            str(output_metrics),
        ]
    )

    metrics_txt = (PROJECT_ROOT / output_metrics).read_text(encoding="utf-8", errors="replace")
    bleu, chrf = _parse_metrics(metrics_txt)

    print("\n=== FULL BASELINE SUMMARY ===")
    print(f"hyp: {output_hyp}")
    print(f"metrics: {output_metrics}")
    print(f"inference_time_s: {inf_time:.2f}")
    print(f"sentences: {lines}")
    print(f"sent_per_s: {speed:.2f}")
    print(f"BLEU: {bleu or '(brak)'}")
    print(f"chrF: {chrf or '(brak)'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

