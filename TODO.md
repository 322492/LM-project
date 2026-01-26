# TODO (checklista) – projekt „Modele językowe”

Poniższa lista jest podzielona na etapy. Elementy, których nie da się ustalić na tym etapie, są celowo zapisane jako TODO.

## Decyzje wstępne / organizacja
- [x] Uzupełnić **skład zespołu** (imiona/nazwiska).
- [ ] Ustalić role / odpowiedzialności (kto prowadzi dane, kto trening, kto raport).
- [ ] Ustalić format oddania (repo + raport + prezentacja) i wymagania prowadzącego (zakres, minimalne elementy).
- [ ] Zdefiniować minimalny zakres eksperymentów (ile konfiguracji treningu / ile wariantów danych) i co jest „must-have”.
- [ ] Ustalić konwencje w repo (struktura katalogów, nazwy plików, sposób zapisu wyników).
- [x] Wprowadzić centralny plik konfiguracji `configs/default.toml` (ścieżki/parametry) oraz zasadę: CLI nadpisuje config.
- [ ] Określić i zanotować **budżet obliczeniowy** jako plan testów:
  - [ ] Czy dostępne jest GPU? (lokalnie / uczelniane / chmura)
  - [ ] Jeśli tylko CPU: czy trening jest wykonalny w czasie projektu?
  - [ ] Pomiary: czas/epoka, maks. batch/seq length, zużycie VRAM/RAM

## Dane
- [x] Pozyskać pary równoległe **EN–PL dla Biblii** (OPUS: Bible-uedin; format Moses + XML).
- [ ] Sprawdzić licencję/warunki użycia danych i zanotować je w repo.
- [x] Wykonać sanity check korpusu równoległego (Moses .en/.pl: spójność linii, puste segmenty, statystyki długości, losowe próbki) — wynik: **60821** par, puste segmenty: 0, pojedyncze outliery długościowe.
- [x] Sprawdzić duplikaty (exact match) w korpusie: EN, PL oraz par (EN,PL) — duplikaty par ~0.73%.
- [ ] Ustalić schemat podziału danych biblijnych:
  - [ ] train/valid/test (np. na poziomie wersetów/rozdziałów/ksiąg – do decyzji)
  - [ ] zasady, aby nie mieszać bardzo podobnych fragmentów między splitami
- [x] Przygotować splity train/val/test (wariant losowy po wersetach/liniiach): **80/5/15**, deterministycznie (**seed=2137**), zapis do `data/splits_random/`.
- [ ] (PORZUCONE) Wykorzystać `data/raw/bible-uedin.en-pl.xml` do splitów po księgach (wariant porzucony na rzecz prostego losowego splitu po wersetach).
- [ ] Oczyścić dane:
  - [ ] normalizacja znaków, usunięcie pustych linii, spójne kodowanie
  - [ ] wykrycie duplikatów i ew. usunięcie
  - [ ] filtrowanie skrajnie długich par / podejrzanych par (heurystyki do decyzji)
- [ ] Zweryfikować jakość dopasowania równoległego (czy EN i PL są rzeczywiście odpowiadającymi sobie segmentami).
- [ ] Przygotować zestawy testowe out-of-domain:
  - [ ] **teksty współczesne** EN–PL (źródło + licencja + format)
  - [ ] **teksty techniczne** EN–PL (źródło + licencja + format)
  - [ ] Zasada: brak przecieku (żadne fragmenty tych testów nie trafiają do treningu)
- [ ] Udokumentować dokładnie finalne rozmiary zbiorów (liczba segmentów, średnia długość, itp.).

## Model i trening (fine-tuning)
- [x] Wybrać model bazowy (baseline) do tłumaczenia EN→PL: `facebook/nllb-200-distilled-600M` (NLLB, tylko inference).
- [x] Fine-tuning małego modelu (mT5-small) na CPU (sanity + 1 epoka):
  - [x] Konfiguracja: `configs/finetune_cpu.toml`
  - [x] Skrypt treningowy: `scripts/finetune_mt5_cpu.py` (z trybem `--quick` dla smoke testu)
  - [x] Skrypt ewaluacji: `scripts/eval_finetuned.py` (automatyczne znajdowanie checkpointu)
  - [x] Uruchomienie sanity run (tryb `--quick`: 2000/200/200 par, 150 kroków) — ✅ zakończone
  - [ ] Uruchomienie pełnego treningu (1 epoka na pełnym zbiorze: 48656/3041 par)
- [ ] Ustalić sposób tokenizacji/segmentacji zgodny z wybranym modelem (bez zmian „na ślepo”).
- [ ] Przygotować pipeline treningowy:
  - [ ] konfiguracja hiperparametrów (learning rate, batch size, max length, liczba epok, warmup, itp.)
  - [ ] walidacja w trakcie treningu (co i jak często liczymy)
  - [ ] zapisywanie checkpointów + możliwość wznowienia treningu
- [ ] Ustalić strategię ograniczeń obliczeniowych (np. mniejszy model / krótszy max length / mniejszy batch) jako jawny kompromis.
- [ ] Zrobić „sanity run” na małym wycinku danych (czy pipeline działa end-to-end).
- [ ] Uruchomić trening właściwy + zapisać:
  - [ ] konfigurację (plik + commit hash)
  - [ ] logi (czas, loss, metryki walidacyjne)
  - [ ] checkpoint końcowy + informację, który checkpoint jest „najlepszy” wg walidacji

## Ewaluacja (in-domain i out-of-domain)
- [x] Zdefiniować metryki automatyczne: **BLEU i chrF** (sacrebleu).
- [x] Ustalić baseline do porównania:
  - [x] **Baseline (NLLB)**: model bazowy **przed** fine-tuningiem (inference tylko)
  - [x] **Fine-tuned (mT5-small)**: model po fine-tuningu na danych biblijnych (tryb quick: ✅ zakończony)
- [x] Baseline inference (EN→PL) na zbiorze testowym: `outputs/baseline/full_test.hyp.pl`.
- [x] Ewaluacja baseline (BLEU + opcjonalnie chrF): `outputs/baseline/full_test.metrics.txt`.
- [x] Ewaluacja fine-tuned (quick mode) na subset testowy: `outputs/finetuned/mt5_small_quick/metrics.txt`.
- [ ] Przeprowadzić ewaluację na:
  - [ ] Biblia (test in-domain) — pełny zbiór testowy (9124 par)
  - [ ] Teksty współczesne (test out-of-domain) — wymaga przygotowania danych
  - [ ] Teksty techniczne (test out-of-domain) — wymaga przygotowania danych
- [ ] Test generalizacji: porównać wyniki Biblia (in-domain) vs teksty współczesne i techniczne (out-of-domain).
- [ ] Zebrać wyniki w powtarzalnym formacie (tabelka + pliki wynikowe).
- [ ] Przygotować próbkę jakościową (kilkanaście–kilkadziesiąt przykładów) do analizy błędów.

## Analiza wyników
- [ ] Porównać baseline vs fine-tuned (po metrykach i na przykładach) — wymaga pełnego treningu i ewaluacji.
- [ ] Porównać: baseline vs po fine-tuningu (oddzielnie dla in-domain i out-of-domain).
- [ ] Sprawdzić, czy fine-tuning poprawił Biblię kosztem degradacji na innych domenach (lub odwrotnie) – opisać wprost.
- [ ] Przeprowadzić krótką analizę błędów:
  - [ ] nazwy własne, archaizmy, składnia, długość zdań, terminologia techniczna, itp.
- [ ] Zanotować ograniczenia eksperymentu (dane, compute, czas, metryki, możliwe biasy).

## Raport / prezentacja
- [ ] Spisać opis danych (źródła, licencje, preprocessing, splity, statystyki).
- [ ] Spisać opis modelu i treningu (co fine-tunowano i jak, konfiguracja, compute).
- [ ] Opisać metody ewaluacji (zbiory testowe, metryki, procedura).
- [ ] Wstawić wyniki (tabele + krótkie komentarze) i analizę jakościową.
- [ ] Dodać sekcję „wnioski” + „co byśmy zrobili dalej” (realistycznie, bez deklaracji).
- [ ] Przygotować slajdy: problem → dane → metoda → wyniki → wnioski.

