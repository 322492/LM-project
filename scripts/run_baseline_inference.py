#!/usr/bin/env python3
"""
UWAGA: CAŁOŚĆ TRWA KILKA GODZIN!

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

from config_utils import get_nested, load_toml, pick


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
    log_every: int = 0,
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

    start_t = time.perf_counter()
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

            if log_every and (len(out) % log_every == 0 or len(out) == len(texts)):
                elapsed = time.perf_counter() - start_t
                speed = (len(out) / elapsed) if elapsed > 0 else 0.0
                remaining = len(texts) - len(out)
                eta = (remaining / speed) if speed > 0 else float("inf")
                eta_s = f"{eta:.0f}s" if eta != float("inf") else "inf"
                print(
                    f"[progress] {len(out)}/{len(texts)} sentences | {speed:.2f} sent/s | elapsed {elapsed:.0f}s | ETA {eta_s}",
                    flush=True,
                )

    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Baseline inference EN->PL (CPU) dla NLLB.")
    ap.add_argument("--config", type=Path, default=Path("configs/default.toml"), help="Plik config TOML.")
    ap.add_argument("--model", type=str, default=None, help="Nazwa modelu Hugging Face. (nadpisuje config)")
    ap.add_argument("--src-lang", type=str, default=None, help="Kod jezyka zrodlowego (NLLB). (nadpisuje config)")
    ap.add_argument("--tgt-lang", type=str, default=None, help="Kod jezyka docelowego (NLLB). (nadpisuje config)")
    ap.add_argument("--input", type=Path, default=None, help="Plik wejsciowy EN. (nadpisuje config)")
    ap.add_argument("--output", type=Path, default=None, help="Plik wyjsciowy PL (hyp). (nadpisuje config)")
    ap.add_argument("--batch-size", type=int, default=None, help="Batch size. (nadpisuje config)")
    ap.add_argument(
        "--input-max-length",
        type=int,
        default=None,
        help="Max dlugosc wejscia w tokenach (truncation). (nadpisuje config)",
    )
    ap.add_argument(
        "--max-length",
        type=int,
        default=None,
        help="Max dlugosc wyjscia w tokenach (generate). (nadpisuje config)",
    )
    ap.add_argument(
        "--log-every",
        type=int,
        default=None,
        help="Co ile zdan wypisac log postepu (0 = wylacz). (nadpisuje config)",
    )
    args = ap.parse_args()

    cfg = load_toml(Path(args.config))

    model_name = str(pick(args.model, get_nested(cfg, ["baseline_nllb", "model_name"]), "facebook/nllb-200-distilled-600M"))
    src_lang = str(pick(args.src_lang, get_nested(cfg, ["baseline_nllb", "src_lang"]), "eng_Latn"))
    tgt_lang = str(pick(args.tgt_lang, get_nested(cfg, ["baseline_nllb", "tgt_lang"]), "pol_Latn"))

    input_path = Path(pick(args.input, get_nested(cfg, ["paths", "splits_random_test_en"]), "data/splits_random/test.en"))
    output_path = Path(pick(args.output, get_nested(cfg, ["paths", "baseline_output_pl"]), "outputs/baseline/test.hyp.pl"))

    batch_size = int(pick(args.batch_size, get_nested(cfg, ["baseline_nllb", "batch_size"]), 4))
    input_max_length = int(pick(args.input_max_length, get_nested(cfg, ["baseline_nllb", "input_max_length"]), 256))
    max_length = int(pick(args.max_length, get_nested(cfg, ["baseline_nllb", "max_length"]), 128))
    log_every = int(pick(args.log_every, get_nested(cfg, ["baseline_nllb", "log_every"]), 64))

    device = torch.device("cpu")

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
    except Exception as e:
        print("ERROR: nie udalo sie pobrac/zaladowac modelu lub tokenizera z Hugging Face.")
        print(f"Model: {model_name}")
        print(f"Szczegoly: {type(e).__name__}: {e}")
        print()
        return 2

    src = read_lines(input_path)
    t0 = time.perf_counter()
    hyps = batch_translate(
        model=model,
        tokenizer=tokenizer,
        texts=src,
        batch_size=batch_size,
        input_max_length=input_max_length,
        output_max_length=max_length,
        src_lang=src_lang,
        tgt_lang=tgt_lang,
        log_every=log_every,
    )
    t1 = time.perf_counter()
    write_lines(output_path, hyps)

    print("=== BASELINE INFERENCE (CPU) ===")
    print(f"Model: {model_name}")
    print(f"Source lang: {src_lang}")
    print(f"Target lang: {tgt_lang}")
    print(f"Input: {input_path}")
    print(f"Output: {output_path}")
    print(f"Sentences: {len(src)}")
    print(f"Total inference time [s]: {t1 - t0:.2f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

