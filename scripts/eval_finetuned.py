#!/usr/bin/env python3
"""
Ewaluacja fine-tuned modelu mT5-small na zbiorze testowym.

Generuje tłumaczenia i liczy BLEU/chrF vs referencje.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from config_utils import get_nested, load_toml, pick


def read_lines(path: Path) -> List[str]:
    with path.open("r", encoding="utf-8", errors="replace") as f:
        return [line.rstrip("\n") for line in f]


def write_lines(path: Path, lines: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", errors="replace") as f:
        for line in lines:
            f.write(line + "\n")


def compute_metrics(hyps: List[str], refs: List[str]) -> tuple[str, str | None]:
    """Liczy BLEU i chrF przez sacrebleu."""
    from sacrebleu.metrics import BLEU, CHRF

    bleu = BLEU()
    bleu_score = bleu.corpus_score(hyps, [refs])
    bleu_line = str(bleu_score)

    chrf_line: str | None = None
    try:
        chrf = CHRF()
        chrf_score = chrf.corpus_score(hyps, [refs])
        chrf_line = str(chrf_score)
    except Exception:
        chrf_line = None

    return bleu_line, chrf_line


def main() -> int:
    ap = argparse.ArgumentParser(description="Ewaluacja fine-tuned mT5-small EN->PL.")
    ap.add_argument("--config", type=Path, default=Path("configs/finetune_cpu.toml"), help="Plik config TOML.")
    ap.add_argument("--checkpoint", type=Path, default=None, help="Ścieżka do checkpointu. (nadpisuje config)")
    ap.add_argument("--test-en", type=Path, default=None, help="Plik test.en. (nadpisuje config)")
    ap.add_argument("--test-pl", type=Path, default=None, help="Plik test.pl (referencje). (nadpisuje config)")
    ap.add_argument("--output-hyp", type=Path, default=None, help="Plik wyjściowy z hipotezami. (nadpisuje config)")
    ap.add_argument("--output-metrics", type=Path, default=None, help="Plik wyjściowy z metrykami. (nadpisuje config)")
    ap.add_argument("--batch-size", type=int, default=None, help="Batch size dla generowania. (nadpisuje config)")
    ap.add_argument("--num-beams", type=int, default=None, help="Liczba beamów. (nadpisuje config)")
    ap.add_argument("--max-new-tokens", type=int, default=None, help="Max nowych tokenów. (nadpisuje config)")
    args = ap.parse_args()

    cfg = load_toml(Path(args.config))

    # Wczytaj parametry
    checkpoint_base_dir = Path(
        pick(args.checkpoint, get_nested(cfg, ["finetune_mt5", "output_dir"]), "outputs/finetuned/mt5_small_quick")
    )
    
    # Sprawdź, czy katalog istnieje
    if not checkpoint_base_dir.exists():
        print(f"ERROR: Katalog checkpointu nie istnieje: {checkpoint_base_dir}")
        print("Sprawdź, czy trening został ukończony i checkpoint został zapisany.")
        return 1
    
    # Sprawdź, czy w katalogu jest bezpośrednio model.safetensors (finalny checkpoint)
    # lub są podkatalogi checkpoint-*
    if (checkpoint_base_dir / "model.safetensors").exists() or (checkpoint_base_dir / "pytorch_model.bin").exists():
        # Finalny checkpoint w głównym katalogu
        checkpoint_dir = checkpoint_base_dir
    else:
        # Szukaj checkpoint-* podkatalogów
        checkpoint_dirs = sorted(checkpoint_base_dir.glob("checkpoint-*"), key=lambda x: int(x.name.split("-")[-1]), reverse=True)
        if not checkpoint_dirs:
            print(f"ERROR: Nie znaleziono checkpointu w: {checkpoint_base_dir}")
            print("Sprawdź, czy trening został ukończony i checkpoint został zapisany.")
            print(f"Zawartość katalogu: {list(checkpoint_base_dir.iterdir())}")
            return 1
        # Użyj najnowszego checkpointu (największy numer)
        checkpoint_dir = checkpoint_dirs[0]
        print(f"Znaleziono checkpointy: {[d.name for d in checkpoint_dirs]}")
        print(f"Używam najnowszego: {checkpoint_dir.name}")
        print()
    test_en_path = Path(pick(args.test_en, get_nested(cfg, ["finetune_mt5", "test_en"]), "data/splits_random/test.en"))
    test_pl_path = Path(pick(args.test_pl, get_nested(cfg, ["finetune_mt5", "test_pl"]), "data/splits_random/test.pl"))
    output_hyp_path = Path(
        pick(args.output_hyp, None, checkpoint_dir / "test.hyp.pl")
    )
    output_metrics_path = Path(
        pick(args.output_metrics, None, checkpoint_dir / "metrics.txt")
    )
    batch_size = int(pick(args.batch_size, get_nested(cfg, ["finetune_mt5", "batch_size"]), 2))
    num_beams = int(pick(args.num_beams, get_nested(cfg, ["finetune_mt5", "num_beams"]), 4))
    max_new_tokens = int(pick(args.max_new_tokens, get_nested(cfg, ["finetune_mt5", "max_new_tokens"]), 128))

    print("=== EWALUACJA FINE-TUNED mT5-small ===")
    print(f"checkpoint: {checkpoint_dir}")
    print(f"test_en: {test_en_path}")
    print(f"test_pl: {test_pl_path}")
    print(f"output_hyp: {output_hyp_path}")
    print(f"batch_size: {batch_size} | num_beams: {num_beams} | max_new_tokens: {max_new_tokens}")
    print()

    # Wczytaj dane
    print("Wczytywanie danych testowych...")
    test_en = read_lines(test_en_path)
    test_pl_refs = read_lines(test_pl_path)
    print(f"Liczba zdań: {len(test_en)}")
    print()

    # Wczytaj model i tokenizer
    # Tokenizer z oryginalnego modelu (checkpoint nie zawiera tokenizera)
    # Model z checkpointu (fine-tuned wagi)
    model_name = str(get_nested(cfg, ["finetune_mt5", "model_name"]) or "google/mt5-small")
    
    print(f"Wczytywanie tokenizera z oryginalnego modelu: {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    print(f"Wczytywanie modelu z checkpointu: {checkpoint_dir}...")
    model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint_dir)
    model.eval()
    print("Model i tokenizer wczytane.")
    print()

    # Generowanie tłumaczeń
    print("Generowanie tłumaczeń...")
    hyps = []
    with torch.inference_mode():
        for i in range(0, len(test_en), batch_size):
            batch = test_en[i : i + batch_size]
            if (i + 1) % 100 == 0 or i + batch_size >= len(test_en):
                print(f"  Przetworzono {min(i + batch_size, len(test_en))}/{len(test_en)} zdań...")

            # Tokenizacja
            inputs = tokenizer(
                batch,
                max_length=128,
                truncation=True,
                padding=True,
                return_tensors="pt",
            )

            # Generowanie
            generated = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
                do_sample=False,
            )

            # Decode
            decoded = tokenizer.batch_decode(generated, skip_special_tokens=True)
            hyps.extend(decoded)

    print(f"Wygenerowano {len(hyps)} tłumaczeń.")
    print()

    # Zapisz hipotezy
    write_lines(output_hyp_path, hyps)
    print(f"Zapisano hipotezy do: {output_hyp_path}")
    print()

    # Sprawdź zgodność długości
    if len(hyps) != len(test_pl_refs):
        print(f"WARNING: różna liczba linii: hyps={len(hyps)}, refs={len(test_pl_refs)}")
        n = min(len(hyps), len(test_pl_refs))
        hyps = hyps[:n]
        test_pl_refs = test_pl_refs[:n]
        print(f"Używam pierwszych {n} linii.")
        print()

    # Licz metryki
    print("Liczenie metryk (BLEU + chrF)...")
    bleu_line, chrf_line = compute_metrics(hyps, test_pl_refs)
    print()

    # Zapisz metryki
    metrics_text = f"""=== METRYKI FINE-TUNED mT5-small ===
checkpoint: {checkpoint_dir}
test_en: {test_en_path}
test_pl: {test_pl_path}
output_hyp: {output_hyp_path}
lines: {len(hyps)}

BLEU: {bleu_line}
chrF: {chrf_line if chrf_line else "(niedostępne)"}
"""
    output_metrics_path.parent.mkdir(parents=True, exist_ok=True)
    output_metrics_path.write_text(metrics_text, encoding="utf-8")

    print(metrics_text)
    print(f"Metryki zapisane do: {output_metrics_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
