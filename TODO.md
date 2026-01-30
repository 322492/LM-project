# TODO – stan projektu (wersja końcowa)

Checklista odzwierciedla zrealizowany zakres projektu. Prezentacja (`results/ResultPresentation.pptx`) może być dopracowana oddzielnie.

---

## Zrealizowane

### Organizacja i konfiguracja
- [x] Skład zespołu (Kamil Tasarz, Jakub Kopystiański).
- [x] Centralna konfiguracja: `configs/default.toml`, `configs/finetune_cpu.toml`; zasada: CLI nadpisuje config.
- [x] Wspólny helper: `scripts/config_utils.py`.

### Dane
- [x] Korpus EN–PL (OPUS Bible-uedin): 60821 par, format Moses + XML w `data/raw/`.
- [x] Sanity check (spójność linii, puste segmenty, statystyki długości).
- [x] Sprawdzenie duplikatów par (~0.73%).
- [x] Splity train/val/test: 80/5/15, losowo po wersetach, seed=2137, `data/splits_random/`.
- [x] Zbiory OOD do ewaluacji: `data/evaluation_data/` (Contemporary, TechnicalGeneral, technicalIT, Theology).

### Baseline i modele
- [x] Baseline NLLB: `facebook/nllb-200-distilled-600M`, inference na pełnym teście (9124 par), metryki w `outputs/baseline/full_test.metrics.txt`.
- [x] mT5-small: pipeline fine-tuningu (config, skrypty, tryb `--quick`), ewaluacja `scripts/eval_finetuned.py`.
- [x] Flan-T5-small: fine-tuning 1 epoka (`scripts/finetune_flan_t5_cpu.py`), checkpoint `outputs/finetuned/flan_t5_small_full/checkpoint-3041`.

### Ewaluacja
- [x] Metryki: BLEU i chrF (sacrebleu).
- [x] Baseline Flan-T5 (zero-shot): `scripts/eval_baseline_flat_t5.py --output-dir results/flan-t5-small/baseline` — wyniki w `results/flan-t5-small/baseline/`.
- [x] Ewaluacja fine-tuned Flan-T5 na Bible + OOD — wyniki w `results/flan-t5-small/finetuned/`.
- [x] Zestawienie wyników: `results/Comparison.xlsx`, `results/ResultPresentation.pptx`.

### Skrypty (podsumowanie)
- [x] Dane: `sanity_check_parallel_moses.py`, `check_duplicates_moses.py`, `make_random_splits_moses.py`.
- [x] Baseline NLLB: `run_baseline_inference.py`, `evaluate_baseline.py`, `run_full_baseline_and_eval.py`.
- [x] mT5: `finetune_mt5_cpu.py`, `eval_finetuned.py`.
- [x] Flan-T5: `finetune_flan_t5_cpu.py`, `eval_finetuned_flan_t5.py`, `eval_baseline_flat_t5.py`.
- [x] OOD: `ood_template_builder.py`; dane w `data/evaluation_data/` i `data/ood/`.

---

## Nie w zakresie wersji końcowej

- Pełny trening mT5-small (1 epoka na pełnym zbiorze) — opcjonalny.
- Formalna dokumentacja licencji danych, raport pisemny, rozbudowana analiza błędów — według wymagań oddania.
