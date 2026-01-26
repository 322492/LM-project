# Projekt końcowy – Modele językowe: fine-tuning EN→PL (korpus biblijny)

## Skład zespołu
- Osoba 1: Kamil Tasarz
- Osoba 2: Jakub Kopystiański

## Krótki opis projektu
Celem projektu jest wykonanie **fine-tuningu istniejącego modelu tłumaczeniowego EN→PL** na danych biblijnych (korpus równoległy), a następnie sprawdzenie **generalizacji** na innych typach tekstów (in-domain i out-of-domain).

## USTALENIA NA START
Poniżej znajdują się wyłącznie decyzje już podjęte:
- Projekt jest realizowany w **2 osoby**.
- Temat: **fine-tuning modelu tłumaczeniowego EN→PL** (bez treningu od zera).
- Dane treningowe: **Biblia jako korpus równoległy EN–PL**.
- Testy / ewaluacja obejmują:
  - **Biblia (in-domain)**,
  - **teksty współczesne (out-of-domain)**,
  - **teksty techniczne (out-of-domain)**.

---

## Status danych (sanity check)
- Źródło: **OPUS – Bible-uedin** (korpus równoległy EN–PL).
- Format:
  - **Moses**: `data/raw/bible-uedin.en-pl.en` + `data/raw/bible-uedin.en-pl.pl` (1 linia = 1 para),
  - **XML**: `data/raw/bible-uedin.en-pl.xml` (metadane).
- Liczba par zdań (Moses): **60821**.
- Wyniki sanity check:
  - brak pustych segmentów (EN: 0, PL: 0),
  - długości zdań są stabilne (mediana: EN 23 słowa, PL 18 słów),
  - niewielka liczba outlierów długościowych (np. 13 bardzo długich linii po stronie EN).
- Konkluzja: dane są spójne i nadają się do dalszej pracy.

## Splity danych
- Używamy **losowego splitu 80/5/15 po wersetach (liniach)**, bez deduplikacji i bez filtrowania długości.
- Split jest deterministyczny: **seed=2137**. Pliki wyjściowe: `data/splits_random/{train,val,test}.{en,pl}`.
- Sprawdzenie duplikatów (exact match) pokazało, że duplikaty par (EN,PL) to ok. **0.73%** — ryzyko przecieku przez identyczne pary jest niskie.
- Dlatego wybrano prosty losowy split (zgodny z praktyką z ćwiczeń) jako punkt startowy do dalszych eksperymentów.

## Model bazowy (baseline)
- **Model**: `facebook/nllb-200-distilled-600M` (NLLB-200, multilingual)
- **Użycie**: tylko **inference** (bez fine-tuningu)
- **Uzasadnienie**:
  - poprawny kierunek **EN→PL**,
  - publicznie dostępny model,
  - dobra jakość tłumaczeń,
  - możliwość uruchomienia inference na CPU.
- **Status**: 
  - ✅ Inference zakończony na pełnym zbiorze testowym (9124 par)
  - ✅ Ewaluacja zakończona: metryki w `outputs/baseline/full_test.metrics.txt`
  - ✅ Skrypty: `scripts/run_baseline_inference.py`, `scripts/evaluate_baseline.py`
  - ✅ Automatyczny pipeline: `scripts/run_full_baseline_and_eval.py`

Nie istnieje publicznie dostępny model **OPUS-MT EN→PL** od **Helsinki-NLP**, dlatego jako baseline wybrano model multilingual (NLLB).

## Fine-tuning (CPU, mały model)
- **Model**: `google/mt5-small` (multilingual T5, encoder-decoder)
- **Uzasadnienie wyboru mT5-small**:
  - **mniejszy rozmiar** niż NLLB-200 (możliwy fine-tuning na CPU),
  - **encoder-decoder architecture** (odpowiednia dla seq2seq),
  - **multilingual** (obsługuje EN i PL),
  - **publicznie dostępny** na Hugging Face Hub.
- **Konfiguracja**: `configs/finetune_cpu.toml`
- **Parametry treningu** (CPU-friendly):
  - batch_size=2, gradient_accumulation_steps=8 (efektywny batch=16),
  - max_source_length=128, max_target_length=128,
  - learning_rate=5e-5, num_epochs=1 (sanity + 1 epoka),
  - warmup_ratio=0.03, seed=2137.
- **Skrypty**:
  - `scripts/finetune_mt5_cpu.py` — trening z metrykami BLEU/chrF na walidacji, obsługuje tryb `--quick` (smoke test),
  - `scripts/eval_finetuned.py` — ewaluacja na zbiorze testowym (automatycznie znajduje najlepszy checkpoint).
- **Status**: 
  - ✅ Implementacja zakończona
  - ✅ Tryb `--quick` działa (smoke test: małe subsety, 150 kroków)
  - ✅ Ewaluacja automatyczna po treningu w trybie quick
  - ⏳ Pełny trening (1 epoka na pełnym zbiorze) — do uruchomienia

**Uwaga**: NLLB pozostaje jako **baseline inference** (bez fine-tuningu), mT5-small jest fine-tunowany na danych biblijnych.

### Wyniki (quick mode - smoke test)
- **Dane**: 2000/200/200 par (train/val/test), 150 kroków treningu
- **Metryki na walidacji**: eval_loss=14.52, eval_bleu=0.0, eval_chrf=0.0
- **Metryki na teście** (subset 200 par): BLEU=0.02, chrF=2.68
- **Uwaga**: Niskie metryki są oczekiwane dla smoke testu (mały subset, krótki trening). Pełny trening powinien dać lepsze wyniki.

## Konfiguracja (config)
- Centralny plik ustawień: `configs/default.toml` (ścieżki, seedy, parametry skryptów).
- Skrypty wspierają flagę `--config` oraz zasadę: **CLI nadpisuje wartości z configa**.
- Wspólny helper do wczytywania configa: `scripts/config_utils.py`.

## Out-of-domain test sets
Projekt przewiduje **2 zbiory testowe out-of-domain** do ewaluacji generalizacji:

1. **Contemporary** (teksty współczesne): `data/ood/contemporary.{en,pl}`
   - Docelowo: 200-500 par z tekstów współczesnych (news, blogi, literatura współczesna).
2. **Technical** (teksty techniczne): `data/ood/technical.{en,pl}`
   - Docelowo: 200-500 par z tekstów technicznych (dokumentacja IT, artykuły naukowe, instrukcje).

**Status:** ⚠️ Na razie są to **placeholdery** — wymagają uzupełnienia źródłem danych i licencją.

Szczegóły i checklista do uzupełnienia: **[`data/ood/README.md`](data/ood/README.md)**

Aby wygenerować szablon dokumentacji:
```bash
python scripts/ood_template_builder.py
```

## TODO (checklista)
Szczegółowa checklista projektu jest w pliku: **[`TODO.md`](TODO.md)**.
