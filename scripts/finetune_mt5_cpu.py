#!/usr/bin/env python3
"""
Fine-tuning mT5-small na CPU dla tłumaczenia EN→PL (dane biblijne).

Używa HuggingFace Seq2SeqTrainer z metrykami BLEU/chrF na walidacji.
CPU-only, bez QLoRA/bitsandbytes.

QUICK MODE (--quick):
  Tryb szybkiego sanity checku i debugowania pipeline'u.
  Używa małych subsetów danych (2000/200/200) i krótkiego treningu (150 kroków).
  Służy tylko do weryfikacji, że pipeline działa, NIE do pełnej ewaluacji modelu.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
from pathlib import Path
from typing import List, Tuple

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


def sample_subset_indices(total_size: int, subset_size: int, seed: int) -> Tuple[List[int], List[int]]:
    """
    Losuje deterministycznie indeksy dla subsetu.
    Zwraca (indices, unused_indices).
    """
    rng = random.Random(seed)
    all_indices = list(range(total_size))
    rng.shuffle(all_indices)
    selected = sorted(all_indices[:subset_size])
    unused = sorted(all_indices[subset_size:])
    return selected, unused


def save_indices(path: Path, indices: List[int]) -> None:
    """Zapisuje indeksy do pliku tekstowego (jeden indeks na linię)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for idx in indices:
            f.write(f"{idx}\n")


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
    """Liczy BLEU i chrF na podstawie predictions i labels używając sacrebleu bezpośrednio."""
    import gc
    from sacrebleu.metrics import BLEU

    predictions, labels = eval_pred
    
    # Konwertuj do numpy jeśli potrzeba (może być tensor)
    if hasattr(predictions, "numpy"):
        predictions = predictions.numpy()
    if hasattr(labels, "numpy"):
        labels = labels.numpy()
    
    # Obsłuż różne formaty: numpy array lub lista list
    # Jeśli to numpy array, konwertuj do listy list
    if isinstance(predictions, np.ndarray):
        # Jeśli ma regularny kształt, użyj tolist()
        try:
            predictions = predictions.tolist()
        except (ValueError, TypeError):
            # Jeśli nieregularny kształt, przetwarzaj jako listę
            predictions = [predictions[i].tolist() if hasattr(predictions[i], 'tolist') else list(predictions[i]) for i in range(len(predictions))]
    elif not isinstance(predictions, list):
        # Jeśli to coś innego, spróbuj przekonwertować
        predictions = list(predictions)
    
    if isinstance(labels, np.ndarray):
        try:
            labels = labels.tolist()
        except (ValueError, TypeError):
            labels = [labels[i].tolist() if hasattr(labels[i], 'tolist') else list(labels[i]) for i in range(len(labels))]
    elif not isinstance(labels, list):
        labels = list(labels)

    # Dekoduj pojedynczo, aby uniknąć MemoryError (najbardziej konserwatywne podejście)
    decoded_preds = []
    decoded_labels = []
    
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    
    num_samples = len(predictions)
    
    # Funkcja pomocnicza do bezpiecznej konwersji do int
    def to_int_safe(x):
        """Bezpieczna konwersja do int, obsługująca różne typy."""
        if isinstance(x, list):
            if len(x) == 0:
                return 0
            return to_int_safe(x[0])
        elif isinstance(x, np.ndarray):
            if x.ndim == 0:
                return int(x.item())
            else:
                return int(x.flat[0])
        elif isinstance(x, (np.integer, np.floating)):
            return int(x)
        else:
            try:
                return int(x)
            except (TypeError, ValueError):
                return 0
    
    # Przetwarzaj pojedynczo, aby uniknąć problemów z pamięcią
    for i in range(num_samples):
        try:
            # Pobierz pojedynczy przykład - unikaj kopiowania całych tablic
            if isinstance(predictions, list):
                pred_item = predictions[i]
            else:
                # Jeśli to numpy array, użyj indeksowania bez kopiowania
                pred_item = predictions[i]
            
            if isinstance(labels, list):
                label_item = labels[i]
            else:
                label_item = labels[i]
            
            # Konwertuj do listy intów BEZPOŚREDNIO, unikając pośrednich struktur
            # Dla predictions - dekoduj bezpośrednio z numpy array jeśli możliwe
            if isinstance(pred_item, np.ndarray):
                # Użyj flat iterator, aby uniknąć kopiowania
                pred_seq = [to_int_safe(int(x)) for x in pred_item.flat if int(x) >= 0]
            elif isinstance(pred_item, list):
                pred_seq = [to_int_safe(x) for x in pred_item if to_int_safe(x) >= 0]
            else:
                # Fallback
                try:
                    pred_seq = [to_int_safe(x) for x in iter(pred_item) if to_int_safe(x) >= 0]
                except:
                    pred_seq = [pad_token_id]
            
            if not pred_seq:
                pred_seq = [pad_token_id]
            
            # Dla labels - podobnie
            if isinstance(label_item, np.ndarray):
                label_seq = [to_int_safe(int(x)) if (int(x) != -100 and int(x) >= 0) else pad_token_id for x in label_item.flat]
            elif isinstance(label_item, list):
                label_seq = [to_int_safe(x) if (to_int_safe(x) != -100 and to_int_safe(x) >= 0) else pad_token_id for x in label_item]
            else:
                try:
                    label_seq = [to_int_safe(x) if (to_int_safe(x) != -100 and to_int_safe(x) >= 0) else pad_token_id for x in iter(label_item)]
                except:
                    label_seq = [pad_token_id]
            
            if not label_seq:
                label_seq = [pad_token_id]
            
            # Decode
            decoded_pred = tokenizer.decode(pred_seq, skip_special_tokens=True)
            decoded_preds.append(decoded_pred)
            
            decoded_label = tokenizer.decode(label_seq, skip_special_tokens=True)
            decoded_labels.append(decoded_label)
            
            # Zwolnij referencje
            del pred_item, label_item, pred_seq, label_seq
            
            # Zwolnij pamięć co 10 przykładów (częściej dla małych zbiorów)
            if (i + 1) % 10 == 0:
                gc.collect()
                
        except MemoryError:
            # Jeśli nadal MemoryError, spróbuj jeszcze prostsze podejście
            print(f"WARNING: MemoryError przy przykładzie {i}, używam uproszczonego dekodowania")
            decoded_preds.append("")
            decoded_labels.append("")
            gc.collect()

    # BLEU przez sacrebleu
    bleu = BLEU()
    bleu_score_obj = bleu.corpus_score(decoded_preds, [decoded_labels])
    bleu_score = bleu_score_obj.score

    # chrF (opcjonalnie)
    chrf_score = None
    try:
        from sacrebleu.metrics import CHRF
        chrf = CHRF()
        chrf_score_obj = chrf.corpus_score(decoded_preds, [decoded_labels])
        chrf_score = chrf_score_obj.score
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
    ap.add_argument("--quick", action="store_true", help="Tryb szybki (smoke test): małe subsety, 150 kroków.")
    ap.add_argument("--resume-from-checkpoint", type=str, default=None, 
                    help="Ścieżka do checkpointu do wznowienia treningu (np. 'checkpoint-250' lub 'outputs/finetuned/mt5_small/checkpoint-250'). "
                         "Jeśli None, trainer automatycznie wykryje ostatni checkpoint w output_dir (jeśli istnieje). "
                         "Aby zacząć od nowa, usuń folder output_dir lub użyj innego output_dir.")
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

    # QUICK MODE: nadpisz parametry
    quick_mode = args.quick
    if quick_mode:
        print("=== QUICK MODE ENABLED ===")
        print("(smoke test: małe subsety, krótki trening)")
        print()
        # Parametry subsetów
        train_subset_size = 2000
        val_subset_size = 200
        test_subset_size = 200
        # Parametry treningu
        max_steps = 150
        eval_steps = 50
        save_steps = 50
        max_source_length = 96
        max_target_length = 96
        batch_size = min(batch_size, 2)
        grad_accum_steps = max(grad_accum_steps, 8)
        output_dir = Path("outputs/finetuned/mt5_small_quick")
        # Wczytaj też test set dla ewaluacji
        test_en = Path(pick(None, get_nested(cfg, ["finetune_mt5", "test_en"]), "data/splits_random/test.en"))
        test_pl = Path(pick(None, get_nested(cfg, ["finetune_mt5", "test_pl"]), "data/splits_random/test.pl"))
    else:
        train_subset_size = None
        val_subset_size = None
        test_subset_size = None
        max_steps = None
        test_en = None
        test_pl = None

    # Seed dla powtarzalności
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    print("=== FINE-TUNING mT5-small (CPU) ===")
    if quick_mode:
        print("MODE: QUICK (smoke test)")
    print(f"model: {model_name}")
    print(f"src_lang: {src_lang} | tgt_lang: {tgt_lang}")
    print(f"batch_size: {batch_size} | grad_accum_steps: {grad_accum_steps} | effective_batch: {batch_size * grad_accum_steps}")
    if max_steps is not None:
        print(f"lr: {learning_rate} | max_steps: {max_steps} | warmup_ratio: {warmup_ratio}")
    else:
        print(f"lr: {learning_rate} | epochs: {num_epochs} | warmup_ratio: {warmup_ratio}")
    print(f"max_source_length: {max_source_length} | max_target_length: {max_target_length}")
    print(f"output_dir: {output_dir}")
    print()

    # Wczytaj dane
    print("Wczytywanie danych...")
    train_data_full = load_parallel_data(train_en, train_pl)
    val_data_full = load_parallel_data(val_en, val_pl)
    
    # QUICK MODE: losuj subsety
    if quick_mode:
        print(f"QUICK MODE: losowanie subsetów (seed={seed})...")
        # Train subset
        train_indices, _ = sample_subset_indices(len(train_data_full), train_subset_size, seed)
        train_data = [train_data_full[i] for i in train_indices]
        save_indices(output_dir / "train.indices.txt", train_indices)
        print(f"  train: {len(train_data)}/{len(train_data_full)} (zapisano indeksy do {output_dir / 'train.indices.txt'})")
        
        # Val subset
        val_indices, _ = sample_subset_indices(len(val_data_full), val_subset_size, seed + 1)
        val_data = [val_data_full[i] for i in val_indices]
        save_indices(output_dir / "val.indices.txt", val_indices)
        print(f"  val: {len(val_data)}/{len(val_data_full)} (zapisano indeksy do {output_dir / 'val.indices.txt'})")
        
        # Test subset (dla późniejszej ewaluacji)
        test_data_full = load_parallel_data(test_en, test_pl)
        test_indices, _ = sample_subset_indices(len(test_data_full), test_subset_size, seed + 2)
        test_data = [test_data_full[i] for i in test_indices]
        save_indices(output_dir / "test.indices.txt", test_indices)
        print(f"  test: {len(test_data)}/{len(test_data_full)} (zapisano indeksy do {output_dir / 'test.indices.txt'})")
        print()
    else:
        train_data = train_data_full
        val_data = val_data_full
        test_data = None
        test_indices = None
    
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
    training_args_dict = {
        "output_dir": str(output_dir),
        "per_device_train_batch_size": batch_size,
        "per_device_eval_batch_size": batch_size,
        "gradient_accumulation_steps": grad_accum_steps,
        "learning_rate": learning_rate,
        "warmup_ratio": warmup_ratio,
        "logging_steps": logging_steps,
        "eval_steps": eval_steps,
        "save_steps": save_steps,
        "eval_strategy": "steps",
        "save_strategy": "steps",
        "load_best_model_at_end": True,
        "metric_for_best_model": "bleu",
        "greater_is_better": True,
        "seed": seed,
        "fp16": False,  # CPU-only
        "dataloader_num_workers": 0,  # CPU-friendly
        "report_to": "none",  # bez wandb/tensorboard
    }
    
    # QUICK MODE: użyj max_steps zamiast num_train_epochs
    if quick_mode:
        training_args_dict["max_steps"] = max_steps
    else:
        training_args_dict["num_train_epochs"] = num_epochs
    
    training_args = Seq2SeqTrainingArguments(**training_args_dict)

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
    
    # Obsługa wznowienia z checkpointu
    resume_from_checkpoint = args.resume_from_checkpoint
    if resume_from_checkpoint is None:
        # Automatyczne wykrywanie: jeśli w output_dir jest checkpoint, wznów z niego
        if output_dir.exists():
            checkpoint_dirs = sorted([d for d in output_dir.iterdir() if d.is_dir() and d.name.startswith("checkpoint-")], 
                                      key=lambda x: int(x.name.split("-")[1]) if len(x.name.split("-")) > 1 and x.name.split("-")[1].isdigit() else 0, 
                                      reverse=True)
            if checkpoint_dirs:
                resume_from_checkpoint = str(checkpoint_dirs[0])
                print(f"Wykryto checkpoint: {resume_from_checkpoint}")
                print("Trening zostanie wznowiony z tego checkpointu.")
                print("(Aby zacząć od nowa, usuń folder output_dir lub użyj --resume-from-checkpoint '')")
            else:
                print("Brak checkpointów - rozpoczynam trening od początku.")
        else:
            print("Brak checkpointów (output_dir nie istnieje) - rozpoczynam trening od początku.")
    elif resume_from_checkpoint == "":
        # Pusty string = wymuszenie treningu od początku
        resume_from_checkpoint = None
        print("Wymuszono trening od początku (--resume-from-checkpoint='').")
    else:
        # Jawnie podana ścieżka
        if not Path(resume_from_checkpoint).exists():
            print(f"UWAGA: Checkpoint '{resume_from_checkpoint}' nie istnieje!")
            print("Trening rozpocznie się od początku.")
            resume_from_checkpoint = None
        else:
            print(f"Wznawiam trening z checkpointu: {resume_from_checkpoint}")
    
    print()
    train_result = trainer.train(resume_from_checkpoint=resume_from_checkpoint)
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
    
    # QUICK MODE: automatyczna ewaluacja na teście
    if quick_mode:
        print()
        print("=== QUICK MODE: Automatyczna ewaluacja na teście ===")
        
        # Zapisz subset testowy do plików tymczasowych
        test_en_subset_path = output_dir / "test_subset.en"
        test_pl_subset_path = output_dir / "test_subset.pl"
        test_en_subset_path.parent.mkdir(parents=True, exist_ok=True)
        
        with test_en_subset_path.open("w", encoding="utf-8") as f:
            for item in test_data:
                f.write(item["translation"]["en"] + "\n")
        with test_pl_subset_path.open("w", encoding="utf-8") as f:
            for item in test_data:
                f.write(item["translation"]["pl"] + "\n")
        
        # Uruchom skrypt ewaluacji
        eval_script = Path(__file__).parent / "eval_finetuned.py"
        eval_cmd = [
            sys.executable,  # użyj tego samego interpretera Python
            str(eval_script),
            "--checkpoint", str(output_dir),
            "--test-en", str(test_en_subset_path),
            "--test-pl", str(test_pl_subset_path),
            "--output-hyp", str(output_dir / "test.hyp.pl"),
            "--output-metrics", str(output_dir / "metrics.txt"),
            "--batch-size", str(batch_size),
        ]
        
        print(f"Uruchamiam: {' '.join(eval_cmd)}")
        try:
            result = subprocess.run(eval_cmd, check=True, capture_output=False, text=True)
            print("Ewaluacja zakończona.")
        except subprocess.CalledProcessError as e:
            print(f"BŁĄD podczas ewaluacji: {e}")
            print("Możesz uruchomić ewaluację ręcznie później.")
        except Exception as e:
            print(f"BŁĄD podczas ewaluacji: {e}")
            print("Możesz uruchomić ewaluację ręcznie później.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
