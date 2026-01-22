# TODO (checklista) – projekt „Modele językowe”

Poniższa lista jest podzielona na etapy. Elementy, których nie da się ustalić na tym etapie, są celowo zapisane jako TODO.

## Decyzje wstępne / organizacja
- [x] Uzupełnić **skład zespołu** (imiona/nazwiska).
- [ ] Ustalić role / odpowiedzialności (kto prowadzi dane, kto trening, kto raport).
- [ ] Ustalić format oddania (repo + raport + prezentacja) i wymagania prowadzącego (zakres, minimalne elementy).
- [ ] Zdefiniować minimalny zakres eksperymentów (ile konfiguracji treningu / ile wariantów danych) i co jest „must-have”.
- [ ] Ustalić konwencje w repo (struktura katalogów, nazwy plików, sposób zapisu wyników).
- [ ] Określić i zanotować **budżet obliczeniowy** jako plan testów:
  - [ ] Czy dostępne jest GPU? (lokalnie / uczelniane / chmura)
  - [ ] Jeśli tylko CPU: czy trening jest wykonalny w czasie projektu?
  - [ ] Pomiary: czas/epoka, maks. batch/seq length, zużycie VRAM/RAM

## Dane
- [ ] Pozyskać pary równoległe **EN–PL dla Biblii** (źródło + format plików).
- [ ] Sprawdzić licencję/warunki użycia danych i zanotować je w repo.
- [ ] Ustalić schemat podziału danych biblijnych:
  - [ ] train/valid/test (np. na poziomie wersetów/rozdziałów/ksiąg – do decyzji)
  - [ ] zasady, aby nie mieszać bardzo podobnych fragmentów między splitami
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
- [ ] Wybrać bazowy **model tłumaczeniowy EN→PL** do fine-tuningu (kryteria: dostępność, rozmiar, wykonalność obliczeniowa).
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
- [ ] Zdefiniować metryki automatyczne (np. BLEU/chrF lub inne – do decyzji) i sposób ich liczenia.
- [ ] Ustalić baseline do porównania:
  - [ ] model bazowy **przed** fine-tuningiem
  - [ ] ewentualnie prosta dodatkowa kontrola (np. inny checkpoint / inna konfiguracja) – jeśli czas pozwoli
- [ ] Przeprowadzić ewaluację na:
  - [ ] Biblia (test in-domain)
  - [ ] Teksty współczesne (test out-of-domain)
  - [ ] Teksty techniczne (test out-of-domain)
- [ ] Zebrać wyniki w powtarzalnym formacie (tabelka + pliki wynikowe).
- [ ] Przygotować próbkę jakościową (kilkanaście–kilkadziesiąt przykładów) do analizy błędów.

## Analiza wyników
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

