#!/usr/bin/env python3
"""
Entry point dla Google Colab - cienki wrapper nad finetune_mt5_cpu.py.

Wypisuje informacje o środowisku (GPU, wersje bibliotek) i wywołuje główny skrypt.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import transformers

# Import głównego skryptu (relatywny import)
script_dir = Path(__file__).parent
sys.path.insert(0, str(script_dir))
from finetune_mt5_cpu import main as finetune_main


def print_env_info() -> None:
    """Wypisuje informacje o środowisku."""
    print("=== ENVIRONMENT INFO ===")
    print(f"PyTorch version: {torch.__version__}")
    print(f"Transformers version: {transformers.__version__}")
    
    if torch.cuda.is_available():
        print(f"CUDA available: YES")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    else:
        print(f"CUDA available: NO (using CPU)")
    print()


def colab_main() -> int:
    """Główna funkcja wrappera dla Colab."""
    ap = argparse.ArgumentParser(
        description="Colab entry point dla fine-tuningu mT5-small EN->PL.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Przykłady użycia:
  # Quick mode (smoke test)
  python scripts/colab_finetune_entry.py --quick
  
  # Pełny trening
  python scripts/colab_finetune_entry.py --epochs 1
  
  # Z własnym katalogiem danych
  python scripts/colab_finetune_entry.py --data-dir /content/drive/MyDrive/data
        """
    )
    
    # Podstawowe flagi - reszta jest przekazywana do finetune_mt5_cpu.py
    ap.add_argument("--quick", action="store_true", help="Quick mode (smoke test)")
    ap.add_argument("--data-dir", type=Path, default=Path("data/splits_random"), 
                    help="Katalog z danymi (default: data/splits_random)")
    ap.add_argument("--output-dir", type=Path, default=None,
                    help="Katalog wyjściowy (default: outputs/finetuned/mt5_small_full)")
    ap.add_argument("--device", type=str, default="auto", choices=["cpu", "cuda", "auto"],
                    help="Urządzenie: cpu, cuda, lub auto (default: auto)")
    
    # Pozwól na przekazanie pozostałych argumentów do finetune_mt5_cpu.py
    args, unknown_args = ap.parse_known_args()
    
    # Wypisz info o środowisku
    print_env_info()
    
    # Przygotuj argumenty dla finetune_mt5_cpu.py
    finetune_args = []
    if args.quick:
        finetune_args.append("--quick")
    if args.data_dir:
        finetune_args.extend(["--data-dir", str(args.data_dir)])
    if args.output_dir:
        finetune_args.extend(["--output-dir", str(args.output_dir)])
    if args.device:
        finetune_args.extend(["--device", args.device])
    
    # Dodaj nieznane argumenty (np. --epochs, --batch-size, etc.)
    finetune_args.extend(unknown_args)
    
    # Zapisz oryginalne sys.argv i ustaw nowe dla argparse w finetune_mt5_cpu.py
    original_argv = sys.argv
    try:
        sys.argv = ["colab_finetune_entry.py"] + finetune_args
        return finetune_main()
    finally:
        sys.argv = original_argv


if __name__ == "__main__":
    raise SystemExit(colab_main())
