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
- **Parametry treningu** (CPU-friendly, zweryfikowane na podstawie quick mode):
  - batch_size=2, gradient_accumulation_steps=8 (efektywny batch=16),
  - max_source_length=96, max_target_length=96 (zmniejszone z 128 dla stabilności pamięciowej),
  - learning_rate=5e-5, num_epochs=1 (sanity + 1 epoka),
  - warmup_ratio=0.03, seed=2137,
  - eval_steps=250, save_steps=250 (częstsze zapisywanie checkpointów),
  - save_total_limit=3 (automatyczne usuwanie starych checkpointów, oszczędza miejsce na dysku).
- **Skrypty**:
  - `scripts/finetune_mt5_cpu.py` — trening z metrykami BLEU/chrF na walidacji, obsługuje tryb `--quick` (smoke test),
  - `scripts/eval_finetuned.py` — ewaluacja na zbiorze testowym (automatycznie znajduje najlepszy checkpoint).
- **Status**: implementacja zakończona; tryb `--quick` (smoke test) i ewaluacja działają. Pełny trening (1 epoka na pełnym zbiorze) pozostaje opcjonalny.

**Uwaga**: NLLB = baseline inference (bez fine-tuningu). mT5-small = model do fine-tuningu na danych biblijnych (quick run zrealizowany).

### mT5-small – wyniki quick mode (smoke test)
- Dane: 2000/200/200 par (train/val/test), 150 kroków. Metryki na teście (subset 200 par): BLEU=0.02, chrF=2.68.

## Flan-T5-small (eksperyment główny – zakończony)
- **Model**: `google/flan-t5-small` (fine-tuning 1 epoka na danych biblijnych).
- **Skrypty**: `scripts/finetune_flan_t5_cpu.py`, `scripts/eval_finetuned_flan_t5.py` (ewaluacja fine-tuned), `scripts/eval_baseline_flat_t5.py` (baseline zero-shot: ten sam model bez fine-tuningu, `--output-dir results/flan-t5-small/baseline`).
- **Wyniki**: `results/flan-t5-small/baseline/` i `results/flan-t5-small/finetuned/` — BLEU/chrF na Bible (in-domain) oraz Contemporary, TechnicalGeneral, technicalIT, Theology (OOD).

## Konfiguracja (config)
- Centralny plik ustawień: `configs/default.toml` (ścieżki, seedy, parametry skryptów).
- Skrypty wspierają flagę `--config` oraz zasadę: **CLI nadpisuje wartości z configa**.
- Wspólny helper do wczytywania configa: `scripts/config_utils.py`.

## Out-of-domain test sets
Projekt przewiduje zbiory testowe out-of-domain do ewaluacji generalizacji. Używane są:
- **data/evaluation_data/** — Contemporary, TechnicalGeneral, technicalIT, Theology (EN/PL, gotowe do ewaluacji).
- **data/ood/** — placeholdery (contemporary, technical); szczegóły w [`data/ood/README.md`](data/ood/README.md). Szablon: `python scripts/ood_template_builder.py`.

## Wyniki i prezentacja
- **results/flan-t5-small/** — wyniki Flan-T5-small: `baseline/` (zero-shot) i `finetuned/` (checkpoint-3041) na Bible + OOD.
- **results/Comparison.xlsx** — zestawienie BLEU/chrF (baseline vs fine-tuned).
- **results/ResultPresentation.pptx** — prezentacja wyników projektu.

Szczegółowa checklista (stan projektu): **[`TODO.md`](TODO.md)**.
