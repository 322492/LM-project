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

## Model bazowy
- model: `facebook/nllb-200-distilled-600M`
- uzasadnienie:
  - poprawny kierunek **EN→PL**,
  - publicznie dostępny model,
  - dobra jakość tłumaczeń,
  - możliwość uruchomienia inference na CPU.

Nie istnieje publicznie dostępny model **OPUS-MT EN→PL** od **Helsinki-NLP**, dlatego jako baseline wybrano model multilingual (NLLB).

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
