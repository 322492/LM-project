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
import json
import random
import time
from datetime import datetime
from pathlib import Path
from typing import List, Sequence, Tuple

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

def write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def batch_translate(
    model: AutoModelForSeq2SeqLM,
    tokenizer: AutoTokenizer,
    texts: Sequence[str],
    batch_size: int,
    input_max_length: int,
    max_new_tokens: int,
    num_beams: int,
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
    with torch.inference_mode():
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
                max_new_tokens=int(max_new_tokens),
                num_beams=int(num_beams),
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


def _select_indices(n: int, max_sentences: int, sample: bool, seed: int) -> List[int]:
    """
    Zwraca listę indeksów 0-based wybranych do inference.
    Zasada:
    - max_sentences <= 0 => wszystkie zdania
    - sample=False => pierwsze max_sentences
    - sample=True  => losowa próbka bez powtórzeń, ale posortowana rosnąco (zachowanie kolejności wyjściowej)
    """
    if n <= 0:
        return []
    if max_sentences <= 0:
        return list(range(n))

    k = min(max_sentences, n)
    if not sample:
        return list(range(k))

    rng = random.Random(seed)
    picked = rng.sample(range(n), k=k)
    return sorted(picked)


def main() -> int:
    ap = argparse.ArgumentParser(description="Baseline inference EN->PL (CPU) dla NLLB.")
    ap.add_argument("--config", type=Path, default=Path("configs/default.toml"), help="Plik config TOML.")
    ap.add_argument("--model", type=str, default=None, help="Nazwa modelu Hugging Face. (nadpisuje config)")
    ap.add_argument("--src-lang", type=str, default=None, help="Kod jezyka zrodlowego (NLLB). (nadpisuje config)")
    ap.add_argument("--tgt-lang", type=str, default=None, help="Kod jezyka docelowego (NLLB). (nadpisuje config)")
    ap.add_argument("--input", type=Path, default=None, help="Plik wejsciowy EN. (nadpisuje config)")
    ap.add_argument("--output", type=Path, default=None, help="Plik wyjsciowy PL (hyp). (nadpisuje domyslne nazwy i config)")
    ap.add_argument("--batch-size", type=int, default=4, help="Batch size (domyslnie: 4; nadpisuje config)")
    ap.add_argument(
        "--input-max-length",
        type=int,
        default=256,
        help="Max dlugosc wejscia w tokenach (truncation). (domyslnie: 256; nadpisuje config)",
    )
    ap.add_argument("--max-new-tokens", type=int, default=64, help="Max new tokens (generate). (domyslnie: 64)")
    ap.add_argument("--num-beams", type=int, default=1, help="Liczba belek (domyslnie: 1 = greedy decoding)")
    ap.add_argument(
        "--max-sentences",
        type=int,
        default=500,
        help="Limit liczby zdan do przetworzenia (0 = caly plik). Domyslnie: 500.",
    )
    ap.add_argument(
        "--sample",
        action="store_true",
        help="Jesli ustawione, wybiera losowa probke zdan (bez powtorzen) zamiast brac pierwsze N.",
    )
    ap.add_argument("--seed", type=int, default=123, help="Seed do probkowania (domyslnie: 123).")
    ap.add_argument(
        "--save-ref-subset",
        action="store_true",
        default=None,
        help="Zapisz dopasowane referencje obok hyps (quick run). Jeśli nie podasz, quick=True, full=False.",
    )
    ap.add_argument(
        "--no-save-ref-subset",
        action="store_true",
        default=None,
        help="Nie zapisuj dopasowanych referencji obok hyps.",
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
    refs_full_path = Path(pick(None, get_nested(cfg, ["paths", "splits_random_test_pl"]), "data/splits_random/test.pl"))

    batch_size = int(pick(args.batch_size, get_nested(cfg, ["baseline_nllb", "batch_size"]), 4))
    input_max_length = int(pick(args.input_max_length, get_nested(cfg, ["baseline_nllb", "input_max_length"]), 256))
    max_new_tokens = int(pick(args.max_new_tokens, get_nested(cfg, ["baseline_nllb", "max_new_tokens"]), 64))
    num_beams = int(pick(args.num_beams, get_nested(cfg, ["baseline_nllb", "num_beams"]), 1))
    max_sentences = int(pick(args.max_sentences, get_nested(cfg, ["baseline_nllb", "max_sentences"]), 500))
    sample = bool(pick(args.sample, get_nested(cfg, ["baseline_nllb", "sample"]), False))
    seed = int(pick(args.seed, get_nested(cfg, ["baseline_nllb", "seed"]), 123))
    log_every = int(pick(args.log_every, get_nested(cfg, ["baseline_nllb", "log_every"]), 64))

    quick = max_sentences > 0

    # Domyślne ścieżki outputu zależne od trybu:
    # - quick: outputs/baseline/quick_{N}.hyp.pl
    # - full:  outputs/baseline/full_test.hyp.pl (z configa paths.baseline_output_pl)
    default_full_out = Path(get_nested(cfg, ["paths", "baseline_output_pl"], "outputs/baseline/full_test.hyp.pl"))
    if quick:
        default_out = Path(f"outputs/baseline/quick_{max_sentences}.hyp.pl")
    else:
        default_out = default_full_out

    output_path = Path(pick(args.output, None, default_out))

    # save-ref-subset domyślnie: quick=True, full=False
    save_ref_subset = True if quick else False
    if args.save_ref_subset:
        save_ref_subset = True
    if args.no_save_ref_subset:
        save_ref_subset = False

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

    src_all = read_lines(input_path)
    indices = _select_indices(len(src_all), max_sentences=max_sentences, sample=sample, seed=seed)
    src = [src_all[i] for i in indices]

    # Sidecar paths obok hyps:
    # - *.indices.txt: jeśli sample=True
    # - *.ref.pl: jeśli quick i save_ref_subset=True
    indices_path = output_path.with_name(f"{output_path.stem}.indices.txt")
    ref_subset_path = output_path.with_name(f"{output_path.stem}.ref.pl")
    meta_path = output_path.with_name(f"{output_path.name}.meta.json")

    if sample:
        indices_path.parent.mkdir(parents=True, exist_ok=True)
        with indices_path.open("w", encoding="utf-8", newline="\n") as f:
            for i in indices:
                f.write(str(i) + "\n")

    if quick and save_ref_subset:
        refs_all = read_lines(refs_full_path)
        refs_subset = [refs_all[i] for i in indices if i < len(refs_all)]
        write_lines(ref_subset_path, refs_subset)

    print("=== BASELINE INFERENCE (CPU) ===")
    print(f"Model: {model_name}")
    print(f"Source lang: {src_lang}")
    print(f"Target lang: {tgt_lang}")
    print(f"Input: {input_path}")
    print(f"Output: {output_path}")
    print(f"Total sentences in file: {len(src_all)}")
    print(f"Selected sentences: {len(src)}")
    print(f"max_sentences: {max_sentences} | sample: {sample} | seed: {seed}")
    print(f"batch_size: {batch_size} | input_max_length: {input_max_length} | max_new_tokens: {max_new_tokens} | num_beams: {num_beams}")
    print(f"quick: {quick} | save_ref_subset: {save_ref_subset}")
    print()

    t0 = time.perf_counter()
    hyps = batch_translate(
        model=model,
        tokenizer=tokenizer,
        texts=src,
        batch_size=batch_size,
        input_max_length=input_max_length,
        max_new_tokens=max_new_tokens,
        num_beams=num_beams,
        src_lang=src_lang,
        tgt_lang=tgt_lang,
        log_every=log_every,
    )
    t1 = time.perf_counter()
    write_lines(output_path, hyps)

    # Metadane runu do powtarzalności (do wykorzystania w ewaluacji/README.txt)
    meta = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "mode": "quick" if quick else "full",
        "model": model_name,
        "src_lang": src_lang,
        "tgt_lang": tgt_lang,
        "batch_size": batch_size,
        "input_max_length": input_max_length,
        "max_new_tokens": max_new_tokens,
        "num_beams": num_beams,
        "max_sentences": max_sentences,
        "sample": sample,
        "seed": seed,
        "input_path": str(input_path),
        "refs_full_path": str(refs_full_path),
        "output_path": str(output_path),
        "ref_subset_path": str(ref_subset_path) if (quick and save_ref_subset) else None,
        "indices_path": str(indices_path) if sample else None,
        "lines_in_file": len(src_all),
        "lines_selected": len(src),
        "inference_time_s": float(t1 - t0),
    }
    write_json(meta_path, meta)

    elapsed = t1 - t0
    speed = (len(src) / elapsed) if elapsed > 0 else 0.0
    print(f"Done. Sentences processed: {len(src)}")
    print(f"Total inference time [s]: {elapsed:.2f}")
    print(f"Speed [sent/s]: {speed:.2f}")

    # 3 pierwsze tłumaczenia do szybkiej kontroli
    print()
    print("== Preview (first 3) ==")
    for j in range(min(3, len(src))):
        print("-" * 60)
        print(f"[idx={indices[j]}] EN: {src[j]}")
        print(f"[idx={indices[j]}] PL: {hyps[j]}")
    if len(src) > 0:
        print("-" * 60)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

