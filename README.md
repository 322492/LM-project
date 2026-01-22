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

## TODO (checklista)
Szczegółowa checklista projektu jest w pliku: **[`TODO.md`](TODO.md)**.
