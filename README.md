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

## Results

| Model | Train domain | Test domain | BLEU | chrF |
|-------|--------------|-------------|------|------|
| NLLB (baseline) | general | Bible (in-domain) | 10.62 | 37.69 |
| NLLB (baseline) | general | OOD | — | — |
| Flan-T5-small (baseline) | general | Bible (in-domain) | 0.07 | 12.29 |
| Flan-T5-small (baseline) | general | OOD (śr. 4 zbiorów) | 0.33 | 13.16 |
| Flan-T5-small (finetuned) | Bible | Bible (in-domain) | 1.70 | 15.82 |
| Flan-T5-small (finetuned) | Bible | OOD (śr. 4 zbiorów) | 0.28 | 8.26 |
| mT5-small (finetuned) | Bible | Bible (subset 200) | 0.02 | 2.68 |
| mT5-small (finetuned) | Bible | OOD | — | — |

Źródło: `outputs/baseline/full_test.metrics.txt`, `results/flan-t5-small/{baseline,finetuned}/*_metrics.txt`, `outputs/finetuned/mt5_small_quick/metrics.txt`. OOD = Contemporary, TechnicalGeneral, technicalIT, Theology.

**Interpretacja:** Fine-tuning na domenie biblijnej wyraźnie poprawia wyniki **in-domain** (Flan-T5: BLEU 0.07→1.70, chrF 12.29→15.82 na Bible). Na zbiorach **out-of-domain** jakość się pogarsza (chrF 13.16→8.26) — model dostosowuje się do stylu treningu i traci na generalizacji. To typowy efekt domenowy: zysk in-domain kosztem OOD.

## Qualitative examples

Przykłady z testu biblijnego (in-domain). Źródło: `data/splits_random/test.{en,pl}`, baseline: `outputs/baseline/full_test.hyp.pl` (NLLB), finetuned: `results/flan-t5-small/finetuned/bible_test.hyp.pl` (Flan-T5-small).

**Przykład 1 (linia 0)**  
EN: *Behold also the ships, which though they be so great, and are driven of fierce winds, yet are they turned about with a very small helm, whithersoever the governor listeth.*  
REF (PL): Oto i okręty, choć tak wielkie są i tęgiemi wiatrami pędzone bywają, wszak i najmniejszym sterem bywają kierowane, gdziekolwiek jest wola sternikowa;  
BASELINE (PL): Zobaczcie też statki, które, choć są wielkie i prowadzone przez gwałtowny wiatr, przemieszczają się z bardzo małym kierownicą, gdzie chce rządca.  
FINETUNED (PL): A tak wszystkich wszystkich, którzy wszystkich jest swoich… (powtórzenia leksykalne)  
Komentarz: Baseline (NLLB) daje płynne PL; finetuned (Flan-T5) produkuje PL w stylu biblijnym z powtórzeniami.

**Przykład 2 (linia 2)**  
EN: *In those days Hezekiah was sick even to death: and he prayed to Yahweh; and he spoke to him, and gave him a sign.*  
REF (PL): I sprzysięgli się przeciw niemu słudzy jego, i zabili go w domu jego.  
BASELINE (PL): W tych dniach Chryzek chory był aż do śmierci, i modlił się do Pana, a On z nim rozmawiał i dał mu znak.  
FINETUNED (PL): A tak dnia dnia Hezekijasz od wity… (powtórzenia, skróty)  
Komentarz: Różnica w stylu (współczesny vs archaiczny) i w stabilności formy.

**Przykład 3 (linia 4)**  
EN: *I have given you every place that the sole of your foot will tread on, as I told Moses.*  
REF (PL): Każde miejsce, po którem deptać będzie stopa nogi waszej, dałem wam, jakom obiecał Mojżeszowi.  
BASELINE (PL): Dałem wam wszelkie miejsce, na które będzie szło pod nogą waszą, jak obiecałem Mojżeszowi.  
FINETUNED (PL): Przeto wszystko nad wszystkiego, którzy wszystko sug sug… (artefakty powtórzeń)  
Komentarz: Finetuned przejmuje archaiczny szyk i słownictwo, z wyraźnymi powtórzeniami.

**Przykład 4 (linia 9)**  
EN: *And it came to pass, when men began to multiply on the face of the earth, and daughters were born unto them,*  
REF (PL): I stało się, gdy się ludzie poczęli rozmnażać na ziemi, a córki się im zrodziły;  
BASELINE (PL): A gdy ludzie zaczęli się liczyć na powierzchni ziemi i urodziły się im córki,  
FINETUNED (PL): A odpowiedzia si, gdy mówi w ziemi ziemi ziemi, i suy suy na ziemi.  
Komentarz: Finetuned powtarza fragmenty (ziemia, odpowiedzia) w stylu zbliżonym do treningu.

**OOD (Contemporary, linie 0 i 5)**  
EN: *That'll do, Donkey. That'll do.* / *The cake is a lie.*  
REF (PL): Wystarczy, osiołku. Wystarczy. / Ciasto to kłamstwo.  
BASELINE (Flan-T5, PL): Donkey. Do not try. / The cake is a lie. (często pozostaje w EN lub krótkie frazy)  
FINETUNED (Flan-T5, PL): Przeto wszystkiego… / Jeli jest te sowa. (formy biblijne, powtórzenia)  
Komentarz: Na OOD baseline (zero-shot) częściej zostaje przy EN lub krótkich frazach; finetuned przenosi styl biblijny i archaiczne formy na tekst współczesny.

## Limitations

- **Budżet obliczeniowy:** Trening i ewaluacja na CPU lub Colab (GPU z limitem czasu); brak systematycznego przeszukiwania konfiguracji.
- **Rozmiar modelu:** Fine-tuning małych modeli (Flan-T5-small, mT5-small); NLLB baseline jest większy i daje wyższe BLEU/chrF na Bible (tabela Results).
- **Długość treningu:** Jedna epoka na danych biblijnych; brak early stopping ani wieloepokowej optymalizacji.
- **Hiperparametry:** Użyte ustawienia z configu (lr, batch, max_length) bez grid search ani walidacji na osobnym dev set; wyniki są reprezentatywne dla tej konfiguracji, nie dla optimum.

## Conclusions

Fine-tuning na korpusie biblijnym **zwiększa wyniki in-domain** (Flan-T5: BLEU i chrF na Bible rosną względem baseline zero-shot) i **obniża wyniki out-of-domain** (chrF na OOD spada). To potwierdza **trade-off specjalizacja vs generalizacja**: model lepiej naśladuje domenę treningu kosztem zdolności do innych typów tekstu. W ramach tego eksperymentu nie rozstrzygamy, czy model uczy się ogólnych zasad tłumaczenia EN→PL, czy głównie stylu i słownictwa biblijnego — wyniki (tabela, przykłady jakościowe) wskazują na silne dostosowanie do stylu treningu.

## Wyniki i prezentacja
- **results/flan-t5-small/** — wyniki Flan-T5-small: `baseline/` (zero-shot) i `finetuned/` (checkpoint-3041) na Bible + OOD.
- **results/Comparison.xlsx** — zestawienie BLEU/chrF (baseline vs fine-tuned).
- **results/ResultPresentation.pptx** — prezentacja wyników projektu.
- Tabela zbiorcza do slajdu: **[`results/slide_results_table.md`](results/slide_results_table.md)**.

Szczegółowa checklista (stan projektu): **[`TODO.md`](TODO.md)**.
