#!/usr/bin/env python3
"""
Baseline inference EN->PL na CPU z uzyciem modelu multilingual:
  facebook/nllb-200-distilled-600M

Skrypt:
- czyta zdania z data/splits_random/test.en,
- wykonuje batch inference na CPU (batch_size=4, jawne max_length),
- ustawia jawnie jezyki: eng_Latn -> pol_Latn,
- uzywa forced_bos_token_id dla jezyka docelowego,
- zapisuje tlumaczenia do outputs/baseline/test.hyp.pl,
- wypisuje logi: nazwa modelu, liczba zdan, czas inference.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import List, Sequence

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


def read_lines(path: Path) -> List[str]:
    with path.open("r", encoding="utf-8", errors="replace") as f:
        return [line.rstrip("\n") for line in f]


def write_lines(path: Path, lines: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for s in lines:
            f.write(s + "\n")


def batch_translate(
    model: AutoModelForSeq2SeqLM,
    tokenizer: AutoTokenizer,
    texts: Sequence[str],
    batch_size: int,
    input_max_length: int,
    output_max_length: int,
    src_lang: str,
    tgt_lang: str,
) -> List[str]:
    """
    Tlumaczenie w batchach. Zachowuje liczbe linii (puste wejscia -> puste wyjscia).
    """
    out: List[str] = []
    model.eval()

    # NLLB: ustawiamy jawnie jezyk zrodlowy na tokenizerze (wymagane dla poprawnych tokenow jezykowych).
    if hasattr(tokenizer, "src_lang"):
        tokenizer.src_lang = src_lang

    # forced_bos_token_id dla jezyka docelowego
    forced_bos_token_id = None
    if hasattr(tokenizer, "lang_code_to_id") and getattr(tokenizer, "lang_code_to_id", None):
        forced_bos_token_id = tokenizer.lang_code_to_id.get(tgt_lang)
    if forced_bos_token_id is None:
        # fallback: sprobojmy konwersji tokenu jezykowego na ID
        try:
            forced_bos_token_id = tokenizer.convert_tokens_to_ids(tgt_lang)
        except Exception:
            forced_bos_token_id = None
    if forced_bos_token_id is None or forced_bos_token_id == tokenizer.unk_token_id:
        raise ValueError(f"Nie udalo sie wyznaczyc forced_bos_token_id dla tgt_lang={tgt_lang!r}.")

    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch = list(texts[start : start + batch_size])

            # Zachowujemy puste linie bez odpalania modelu.
            non_empty = [(i, t) for i, t in enumerate(batch) if t.strip()]
            if not non_empty:
                out.extend([""] * len(batch))
                continue

            idxs, non_empty_texts = zip(*non_empty)
            enc = tokenizer(
                list(non_empty_texts),
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=input_max_length,
            )

            # CPU-only
            enc = {k: v.to("cpu") for k, v in enc.items()}

            gen = model.generate(
                **enc,
                max_length=output_max_length,  # jawnie ustawione
                forced_bos_token_id=int(forced_bos_token_id),
            )
            decoded = tokenizer.batch_decode(gen, skip_special_tokens=True)

            # Wstawiamy przetlumaczone teksty z powrotem na odpowiednie pozycje w batchu.
            batch_out = [""] * len(batch)
            for pos, hyp in zip(idxs, decoded):
                batch_out[pos] = hyp
            out.extend(batch_out)

    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Baseline inference EN->PL (CPU) dla facebook/nllb-200-distilled-600M.")
    ap.add_argument("--model", type=str, default="facebook/nllb-200-distilled-600M", help="Nazwa modelu Hugging Face.")
    ap.add_argument("--src-lang", type=str, default="eng_Latn", help="Kod jezyka zrodlowego (NLLB), domyslnie: eng_Latn.")
    ap.add_argument("--tgt-lang", type=str, default="pol_Latn", help="Kod jezyka docelowego (NLLB), domyslnie: pol_Latn.")
    ap.add_argument("--input", type=Path, default=Path("data/splits_random/test.en"), help="Plik wejsciowy EN.")
    ap.add_argument("--output", type=Path, default=Path("outputs/baseline/test.hyp.pl"), help="Plik wyjsciowy PL (hyp).")
    ap.add_argument("--batch-size", type=int, default=4, help="Batch size (domyslnie: 4).")
    ap.add_argument(
        "--input-max-length",
        type=int,
        default=256,
        help="Max dlugosc wejscia w tokenach (truncation), domyslnie: 256.",
    )
    ap.add_argument(
        "--max-length",
        type=int,
        default=128,
        help="Max dlugosc wyjscia w tokenach (generate), domyslnie: 128.",
    )
    args = ap.parse_args()

    device = torch.device("cpu")

    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        model = AutoModelForSeq2SeqLM.from_pretrained(args.model).to(device)
    except Exception as e:
        print("ERROR: nie udalo sie pobrac/zaladowac modelu lub tokenizera z Hugging Face.")
        print(f"Model: {args.model}")
        print(f"Szczegoly: {type(e).__name__}: {e}")
        print()
        print("Wskazowka: jesli masz blokade sieci/proxy na huggingface.co, pobieranie modeli moze nie dzialac.")
        print("Upewnij sie tez, ze masz zainstalowane: torch + transformers + sentencepiece.")
        return 2

    src = read_lines(args.input)
    t0 = time.perf_counter()
    hyps = batch_translate(
        model=model,
        tokenizer=tokenizer,
        texts=src,
        batch_size=int(args.batch_size),
        input_max_length=int(args.input_max_length),
        output_max_length=int(args.max_length),
        src_lang=str(args.src_lang),
        tgt_lang=str(args.tgt_lang),
    )
    t1 = time.perf_counter()
    write_lines(args.output, hyps)

    print("=== BASELINE INFERENCE (CPU) ===")
    print(f"Model: {args.model}")
    print(f"Source lang: {args.src_lang}")
    print(f"Target lang: {args.tgt_lang}")
    print(f"Input: {args.input}")
    print(f"Output: {args.output}")
    print(f"Sentences: {len(src)}")
    print(f"Total inference time [s]: {t1 - t0:.2f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

