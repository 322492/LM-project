"""
UWAGA: Ten skrypt jest pomocniczy (trening + ewaluacja) i może być kosztowny na CPU.

Zasada projektu: konfiguracja w jednym miejscu (configs/default.toml).
Nie hardcodujemy nazw modeli ani ścieżek splitów.

Domyślny model: facebook/nllb-200-distilled-600M (NLLB, multilingual).
Kierunek: EN -> PL (eng_Latn -> pol_Latn).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import torch
from datasets import Dataset, DatasetDict
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from config_utils import get_nested, load_toml, pick


def read_parallel(en_path: Path, pl_path: Path) -> List[dict]:
    en_lines = en_path.read_text(encoding="utf-8", errors="replace").splitlines()
    pl_lines = pl_path.read_text(encoding="utf-8", errors="replace").splitlines()
    if len(en_lines) != len(pl_lines):
        raise ValueError(f"Mismatched dataset sizes: EN={len(en_lines)} PL={len(pl_lines)}")
    # Ujednolicamy format: translation={en:..., pl:...}
    return [{"translation": {"en": en.strip(), "pl": pl.strip()}} for en, pl in zip(en_lines, pl_lines)]


def nllb_forced_bos_id(tokenizer, tgt_lang: str) -> int:
    forced = None
    if hasattr(tokenizer, "lang_code_to_id") and getattr(tokenizer, "lang_code_to_id", None):
        forced = tokenizer.lang_code_to_id.get(tgt_lang)
    if forced is None:
        forced = tokenizer.convert_tokens_to_ids(tgt_lang)
    if forced is None or forced == tokenizer.unk_token_id:
        raise ValueError(f"Nie udalo sie wyznaczyc forced_bos_token_id dla tgt_lang={tgt_lang!r}.")
    return int(forced)


def main() -> int:
    ap = argparse.ArgumentParser(description="(WIP) Train + evaluate dla EN->PL (NLLB) z configa.")
    ap.add_argument("--config", type=Path, default=Path("configs/default.toml"))
    ap.add_argument("--model", type=str, default=None, help="Nazwa modelu (nadpisuje config)")
    ap.add_argument("--src-lang", type=str, default=None, help="Kod src (nadpisuje config)")
    ap.add_argument("--tgt-lang", type=str, default=None, help="Kod tgt (nadpisuje config)")
    args = ap.parse_args()

    cfg = load_toml(Path(args.config))
    model_name = str(pick(args.model, get_nested(cfg, ["baseline_nllb", "model_name"]), "facebook/nllb-200-distilled-600M"))
    src_lang = str(pick(args.src_lang, get_nested(cfg, ["baseline_nllb", "src_lang"]), "eng_Latn"))
    tgt_lang = str(pick(args.tgt_lang, get_nested(cfg, ["baseline_nllb", "tgt_lang"]), "pol_Latn"))

    train_en = Path(get_nested(cfg, ["paths", "splits_random_train_en"]))
    train_pl = Path(get_nested(cfg, ["paths", "splits_random_train_pl"]))
    val_en = Path(get_nested(cfg, ["paths", "splits_random_val_en"]))
    val_pl = Path(get_nested(cfg, ["paths", "splits_random_val_pl"]))
    test_en = Path(get_nested(cfg, ["paths", "splits_random_test_en"]))
    test_pl = Path(get_nested(cfg, ["paths", "splits_random_test_pl"]))

    data = DatasetDict(
        {
            "train": Dataset.from_list(read_parallel(train_en, train_pl)),
            "validation": Dataset.from_list(read_parallel(val_en, val_pl)),
            "test": Dataset.from_list(read_parallel(test_en, test_pl)),
        }
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if hasattr(tokenizer, "src_lang"):
        tokenizer.src_lang = src_lang
    forced_bos_token_id = nllb_forced_bos_id(tokenizer, tgt_lang=tgt_lang)

    # Szybka sanity informacja (bez trenowania)
    print("=== TRAIN/EVAL (config-driven) ===")
    print(f"model: {model_name}")
    print(f"src_lang: {src_lang} | tgt_lang: {tgt_lang} | forced_bos_token_id: {forced_bos_token_id}")
    print(f"sizes: train={len(data['train'])}, val={len(data['validation'])}, test={len(data['test'])}")
    print("UWAGA: Ten skrypt nie trenuje automatycznie (trening pozostaje do dopracowania/uruchomienia osobno).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())