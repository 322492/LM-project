# Fine-tuning mT5-small w Google Colab

Instrukcja uruchomienia fine-tuningu modelu mT5-small dla tłumaczenia EN→PL w Google Colab (z GPU).

## Wymagania

- Google Colab z aktywowanym GPU (Runtime → Change runtime type → GPU)
- Dane treningowe w formacie: `train.en`, `train.pl`, `val.en`, `val.pl`, `test.en`, `test.pl`

## Instalacja

### 1. Sklonuj repozytorium

```python
!git clone <URL_REPOZYTORIUM>
%cd <NAZWA_REPOZYTORIUM>
```

### 2. Zainstaluj zależności

```python
!pip install torch transformers datasets sacrebleu sentencepiece accelerate evaluate
```

Lub jeśli masz `requirements.txt` w repo (bez CPU-only index):

```python
!pip install -r requirements.txt
```

**UWAGA**: W Colab nie używaj `--extra-index-url https://download.pytorch.org/whl/cpu` - Colab ma już PyTorch z CUDA.

### 3. (Opcjonalnie) Zamontuj Google Drive

Jeśli dane są w Google Drive:

```python
from google.colab import drive
drive.mount('/content/drive')

# Ustaw ścieżkę do danych
DATA_DIR = "/content/drive/MyDrive/path/to/data/splits_random"
```

Jeśli dane są już w repo (po `git clone`):

```python
DATA_DIR = "data/splits_random"  # domyślna ścieżka
```

## Uruchomienie

### Quick mode (smoke test)

Szybki test, żeby sprawdzić, że wszystko działa:

```python
# Z metrykami (może być OOM na GPU)
!python scripts/colab_finetune_entry.py --quick

# Bez metryk podczas treningu (zalecane na GPU, jeśli OOM)
!python scripts/colab_finetune_entry.py --quick --skip-eval-metrics
```

Lub bezpośrednio:

```python
!python scripts/finetune_mt5_cpu.py --quick --device auto
```

**Quick mode używa:**
- 2000/200/200 par (train/val/test)
- 150 kroków treningu
- Output: `outputs/finetuned/mt5_small_quick/`

### Pełny trening

```python
!python scripts/colab_finetune_entry.py --epochs 1
```

Lub z własnym katalogiem danych:

```python
!python scripts/colab_finetune_entry.py \
  --data-dir /content/drive/MyDrive/data/splits_random \
  --epochs 1 \
  --output-dir outputs/finetuned/mt5_small_full
```

**Pełny trening używa:**
- Wszystkie dane (48656/3041 par train/val)
- 1 epoka
- Output: `outputs/finetuned/mt5_small_full/` (domyślnie)

### Zaawansowane opcje

```python
# Własny batch size i learning rate
!python scripts/colab_finetune_entry.py \
  --epochs 1 \
  --batch-size 4 \
  --lr 1e-4

# Wymuszenie CPU (jeśli chcesz przetestować bez GPU)
!python scripts/colab_finetune_entry.py --device cpu --quick

# Wznów trening z checkpointu
!python scripts/colab_finetune_entry.py \
  --resume-from-checkpoint outputs/finetuned/mt5_small_full/checkpoint-500
```

## Struktura danych

Skrypt oczekuje następującej struktury katalogów:

```
data/splits_random/
├── train.en
├── train.pl
├── val.en
├── val.pl
├── test.en
└── test.pl
```

Każdy plik to jedna linia = jedno zdanie.

## Output

Po zakończeniu treningu znajdziesz:

- **Checkpointy**: `outputs/finetuned/mt5_small_full/checkpoint-XXX/`
- **Metryki treningu**: `outputs/finetuned/mt5_small_full/training_metrics.json`
- **Najlepszy model**: automatycznie załadowany na końcu treningu (wg BLEU na walidacji)

## Wznawianie treningu

Skrypt automatycznie wykrywa checkpointy w `output_dir`. Jeśli przerwiesz trening i uruchomisz ponownie z tym samym `--output-dir`, trening zostanie wznowiony z ostatniego checkpointu.

Aby zacząć od nowa:

```python
# Usuń folder output lub użyj innego output_dir
!rm -rf outputs/finetuned/mt5_small_full

# Lub wymuś start od początku
!python scripts/colab_finetune_entry.py \
  --resume-from-checkpoint "" \
  --epochs 1
```

## Troubleshooting

### Ostrzeżenia TensorFlow/XLA (nie są problemem)

Na początku uruchomienia możesz zobaczyć ostrzeżenia typu:
```
Unable to register cuFFT factory: Attempting to register factory...
Unable to register cuDNN factory: Attempting to register factory...
```

**To jest normalne** - Colab ma TensorFlow i PyTorch zainstalowane jednocześnie, co powoduje te ostrzeżenia. Nie wpływają one na trening i można je zignorować.

### Brak GPU

Jeśli Colab nie przydzielił GPU:
1. Runtime → Change runtime type → GPU
2. Sprawdź: `!nvidia-smi`

### Out of Memory (OOM)

Jeśli dostaniesz błąd `CUDA out of memory` podczas ewaluacji:

1. **Najlepsze rozwiązanie: wyłącz metryki podczas treningu**:
```python
!python scripts/colab_finetune_entry.py --quick --skip-eval-metrics
```

To wyłączy obliczanie BLEU/chrF podczas ewaluacji w trakcie treningu (tylko loss). Metryki można policzyć później przez `eval_finetuned.py`.

2. **Zmniejsz batch size**:
```python
!python scripts/colab_finetune_entry.py --batch-size 1 --epochs 1
```

3. **Wyczyść pamięć GPU** przed uruchomieniem:
```python
import torch
torch.cuda.empty_cache()
```

4. **Restart runtime** (Runtime → Restart runtime) - zwolni całą pamięć

**Uwaga**: 
- Skrypt automatycznie zmniejsza `eval_batch_size` na GPU (o połowę w stosunku do `batch_size`), aby uniknąć OOM podczas ewaluacji.
- `--skip-eval-metrics` jest szczególnie przydatne na GPU - metryki można policzyć później przez `scripts/eval_finetuned.py`.

### Błąd importu

Upewnij się, że jesteś w katalogu repo:

```python
import os
print(os.getcwd())
# Jeśli nie jesteś w repo: %cd <NAZWA_REPOZYTORIUM>
```

## Szacowany czas

- **Quick mode**: ~10-15 minut
- **Pełny trening (1 epoka)**: ~30-60 minut na GPU (zależy od typu GPU)

## Notatki

- Skrypt automatycznie używa GPU jeśli dostępne (`--device auto`)
- Mixed precision (fp16) jest włączone automatycznie na GPU
- Checkpointy są zapisywane co 250 kroków (można zmienić w configu)
