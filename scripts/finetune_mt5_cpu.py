#!/usr/bin/env python3
"""
Fine-tuning mT5-small na CPU dla tłumaczenia EN→PL (dane biblijne).

Używa HuggingFace Seq2SeqTrainer z metrykami BLEU/chrF na walidacji.
CPU-only, bez QLoRA/bitsandbytes.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import List

import numpy as np
import torch
from datasets import Dataset
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

from config_utils import get_nested, load_toml, pick


def read_lines(path: Path) -> List[str]:
    with path.open("r", encoding="utf-8", errors="replace") as f:
        return [line.rstrip("\n") for line in f]


def load_parallel_data(en_path: Path, pl_path: Path) -> List[dict]:
    """Wczytuje równoległe dane EN-PL i zwraca listę dict z kluczem 'translation'."""
    en_lines = read_lines(en_path)
    pl_lines = read_lines(pl_path)
    if len(en_lines) != len(pl_lines):
        raise ValueError(f"Mismatched sizes: EN={len(en_lines)} PL={len(pl_lines)}")
    return [{"translation": {"en": en.strip(), "pl": pl.strip()}} for en, pl in zip(en_lines, pl_lines)]


def preprocess_function(examples, tokenizer, max_source_length: int, max_target_length: int, src_lang: str, tgt_lang: str):
    """
    Tokenizuje dane dla seq2seq.
    Dla mT5: source = EN, target = PL.
    
    Przy batched=True, examples to dict z kluczami = nazwy kolumn,
    wartości = listy. examples["translation"] to lista dictów z kluczami "en" i "pl".
    """
    # examples["translation"] to lista dictów: [{"en": "...", "pl": "..."}, ...]
    inputs = [ex[src_lang] for ex in examples["translation"]]
    targets = [ex[tgt_lang] for ex in examples["translation"]]

    # Tokenizacja source (EN)
    model_inputs = tokenizer(
        inputs,
        max_length=max_source_length,
        truncation=True,
        padding=False,
        return_tensors=None,
    )

    # Tokenizacja target (PL) - tylko dla labels
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            targets,
            max_length=max_target_length,
            truncation=True,
            padding=False,
            return_tensors=None,
        )

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


def compute_metrics(eval_pred, tokenizer):
    """Liczy BLEU i chrF na podstawie predictions i labels."""
    try:
        import evaluate
    except ImportError:
        # Fallback dla starszych wersji datasets
        from datasets import load_metric
        evaluate = None

    predictions, labels = eval_pred

    # Decode predictions (skip special tokens)
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)

    # Replace -100 w labels (ignore index) przez pad_token_id
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # BLEU
    if evaluate is not None:
        bleu_metric = evaluate.load("sacrebleu")
        bleu_result = bleu_metric.compute(predictions=decoded_preds, references=[[ref] for ref in decoded_labels])
        bleu_score = bleu_result["score"]
    else:
        bleu_metric = load_metric("sacrebleu")
        bleu_result = bleu_metric.compute(predictions=decoded_preds, references=[[ref] for ref in decoded_labels])
        bleu_score = bleu_result["score"]

    # chrF (opcjonalnie)
    chrf_score = None
    try:
        if evaluate is not None:
            chrf_metric = evaluate.load("chrf")
            chrf_result = chrf_metric.compute(predictions=decoded_preds, references=[[ref] for ref in decoded_labels])
            chrf_score = chrf_result["score"]
        else:
            chrf_metric = load_metric("chrf")
            chrf_result = chrf_metric.compute(predictions=decoded_preds, references=[[ref] for ref in decoded_labels])
            chrf_score = chrf_result["score"]
    except Exception:
        chrf_score = None

    result = {"bleu": round(bleu_score, 4)}
    if chrf_score is not None:
        result["chrf"] = round(chrf_score, 4)

    return result


def main() -> int:
    ap = argparse.ArgumentParser(description="Fine-tuning mT5-small EN->PL na CPU (dane biblijne).")
    ap.add_argument("--config", type=Path, default=Path("configs/finetune_cpu.toml"), help="Plik config TOML.")
    ap.add_argument("--model", type=str, default=None, help="Nazwa modelu. (nadpisuje config)")
    ap.add_argument("--output-dir", type=Path, default=None, help="Katalog wyjściowy. (nadpisuje config)")
    ap.add_argument("--epochs", type=int, default=None, help="Liczba epok. (nadpisuje config)")
    ap.add_argument("--batch-size", type=int, default=None, help="Batch size. (nadpisuje config)")
    ap.add_argument("--lr", type=float, default=None, help="Learning rate. (nadpisuje config)")
    args = ap.parse_args()

    cfg = load_toml(Path(args.config))

    # Wczytaj parametry z configa
    model_name = str(pick(args.model, get_nested(cfg, ["finetune_mt5", "model_name"]), "google/mt5-small"))
    src_lang = str(get_nested(cfg, ["finetune_mt5", "src_lang"]) or "en")
    tgt_lang = str(get_nested(cfg, ["finetune_mt5", "tgt_lang"]) or "pl")

    train_en = Path(pick(None, get_nested(cfg, ["finetune_mt5", "train_en"]), "data/splits_random/train.en"))
    train_pl = Path(pick(None, get_nested(cfg, ["finetune_mt5", "train_pl"]), "data/splits_random/train.pl"))
    val_en = Path(pick(None, get_nested(cfg, ["finetune_mt5", "val_en"]), "data/splits_random/val.en"))
    val_pl = Path(pick(None, get_nested(cfg, ["finetune_mt5", "val_pl"]), "data/splits_random/val.pl"))

    max_source_length = int(pick(None, get_nested(cfg, ["finetune_mt5", "max_source_length"]), 128))
    max_target_length = int(pick(None, get_nested(cfg, ["finetune_mt5", "max_target_length"]), 128))
    batch_size = int(pick(args.batch_size, get_nested(cfg, ["finetune_mt5", "batch_size"]), 2))
    grad_accum_steps = int(pick(None, get_nested(cfg, ["finetune_mt5", "grad_accum_steps"]), 8))
    learning_rate = float(pick(args.lr, get_nested(cfg, ["finetune_mt5", "learning_rate"]), 5.0e-5))
    num_epochs = int(pick(args.epochs, get_nested(cfg, ["finetune_mt5", "num_epochs"]), 1))
    warmup_ratio = float(pick(None, get_nested(cfg, ["finetune_mt5", "warmup_ratio"]), 0.03))
    seed = int(pick(None, get_nested(cfg, ["finetune_mt5", "seed"]), 2137))

    output_dir = Path(pick(args.output_dir, get_nested(cfg, ["finetune_mt5", "output_dir"]), "outputs/finetuned/mt5_small"))
    eval_steps = int(pick(None, get_nested(cfg, ["finetune_mt5", "eval_steps"]), 500))
    save_steps = int(pick(None, get_nested(cfg, ["finetune_mt5", "save_steps"]), 500))
    logging_steps = int(pick(None, get_nested(cfg, ["finetune_mt5", "logging_steps"]), 50))

    # Seed dla powtarzalności
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    print("=== FINE-TUNING mT5-small (CPU) ===")
    print(f"model: {model_name}")
    print(f"src_lang: {src_lang} | tgt_lang: {tgt_lang}")
    print(f"batch_size: {batch_size} | grad_accum_steps: {grad_accum_steps} | effective_batch: {batch_size * grad_accum_steps}")
    print(f"lr: {learning_rate} | epochs: {num_epochs} | warmup_ratio: {warmup_ratio}")
    print(f"max_source_length: {max_source_length} | max_target_length: {max_target_length}")
    print(f"output_dir: {output_dir}")
    print()

    # Wczytaj dane
    print("Wczytywanie danych...")
    train_data = load_parallel_data(train_en, train_pl)
    val_data = load_parallel_data(val_en, val_pl)
    train_dataset = Dataset.from_list(train_data)
    val_dataset = Dataset.from_list(val_data)
    print(f"train: {len(train_dataset)} par | val: {len(val_dataset)} par")
    print()

    # Wczytaj model i tokenizer
    print(f"Wczytywanie modelu i tokenizera: {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    # Preprocessing
    print("Tokenizacja danych...")
    train_dataset = train_dataset.map(
        lambda x: preprocess_function(x, tokenizer, max_source_length, max_target_length, src_lang, tgt_lang),
        batched=True,
        remove_columns=train_dataset.column_names,
    )
    val_dataset = val_dataset.map(
        lambda x: preprocess_function(x, tokenizer, max_source_length, max_target_length, src_lang, tgt_lang),
        batched=True,
        remove_columns=val_dataset.column_names,
    )
    print("Tokenizacja zakończona.")
    print()

    # Data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        return_tensors="pt",
    )

    # Training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum_steps,
        learning_rate=learning_rate,
        warmup_ratio=warmup_ratio,
        logging_steps=logging_steps,
        eval_steps=eval_steps,
        save_steps=save_steps,
        eval_strategy="steps",  # nowsze wersje Transformers używają eval_strategy zamiast evaluation_strategy
        save_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="bleu",
        greater_is_better=True,
        seed=seed,
        fp16=False,  # CPU-only
        dataloader_num_workers=0,  # CPU-friendly
        report_to="none",  # bez wandb/tensorboard
    )

    # Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=lambda eval_pred: compute_metrics(eval_pred, tokenizer),
    )

    # Trening
    print("Rozpoczynam trening...")
    train_result = trainer.train()
    print("Trening zakończony.")
    print()

    # Zapisz finalne metryki
    final_metrics = {
        "train_loss": train_result.training_loss,
        "train_runtime": train_result.metrics.get("train_runtime", 0),
        "train_samples_per_second": train_result.metrics.get("train_samples_per_second", 0),
    }

    # Ewaluacja na walidacji
    print("Ewaluacja na walidacji...")
    eval_metrics = trainer.evaluate()
    final_metrics.update(eval_metrics)

    # Zapisz metryki do pliku
    metrics_path = output_dir / "training_metrics.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(final_metrics, f, indent=2, ensure_ascii=False)

    print(f"\n=== PODSUMOWANIE ===")
    print(f"train_loss: {final_metrics.get('train_loss', 'N/A'):.4f}")
    print(f"eval_bleu: {final_metrics.get('eval_bleu', 'N/A'):.4f}")
    if "eval_chrf" in final_metrics:
        print(f"eval_chrf: {final_metrics.get('eval_chrf', 'N/A'):.4f}")
    print(f"train_runtime: {final_metrics.get('train_runtime', 0):.2f}s")
    print(f"Metryki zapisane do: {metrics_path}")
    print(f"Checkpoint zapisany do: {output_dir}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
