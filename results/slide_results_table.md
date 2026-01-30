# Slajd: Wyniki – efekt domenowy (in-domain vs OOD)

---

## Tabela (pełna)

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

---

## Skrót na slajd (tylko Flan-T5 – efekt domenowy)

| Model | Test domain | BLEU | chrF |
|-------|-------------|------|------|
| Flan-T5-small (baseline) | Bible (in-domain) | 0.07 | 12.29 |
| Flan-T5-small (baseline) | OOD | 0.33 | 13.16 |
| Flan-T5-small (finetuned) | Bible (in-domain) | **1.70** | **15.82** |
| Flan-T5-small (finetuned) | OOD | 0.28 | 8.26 |

**Wniosek:** Fine-tuning poprawia wyniki in-domain (Bible), pogarsza generalizację OOD.

---

## Tekst do slajdu (2–3 zdania)

Fine-tuning na danych biblijnych wyraźnie poprawia wyniki **in-domain** (BLEU i chrF na Bible rosną). Na zbiorach **out-of-domain** (Contemporary, Technical, Theology) jakość spada — model dostosowuje się do domeny treningu i traci na generalizacji. Typowy efekt: zysk in-domain kosztem OOD.

---

# Slajd: Przykłady jakościowe (max 2 przykłady)

---

### Przykład 1 – Bible (in-domain)

| | Tekst |
|---|--------|
| **EN** | Behold also the ships, which though they be so great, and are driven of fierce winds, yet are they turned about with a very small helm, whithersoever the governor listeth. |
| **REF (PL)** | Oto i okręty, choć tak wielkie są i tęgiemi wiatrami pędzone bywają, wszak i najmniejszym sterem bywają kierowane, gdziekolwiek jest wola sternikowa; |
| **BASELINE (NLLB)** | Zobaczcie też statki, które, choć są wielkie i prowadzone przez gwałtowny wiatr, przemieszczają się z bardzo małym kierownicą, gdzie chce rządca. |
| **FINETUNED (Flan-T5)** | A tak wszystkich wszystkich… (PL w stylu biblijnym, powtórzenia leksykalne) |

Komentarz: Baseline daje płynne PL; finetuned produkuje PL w stylu treningu z powtórzeniami.

---

### Przykład 2 – OOD (Contemporary)

| | Tekst |
|---|--------|
| **EN** | The cake is a lie. |
| **REF (PL)** | Ciasto to kłamstwo. |
| **BASELINE (Flan-T5)** | The cake is a lie. (pozostaje w EN) |
| **FINETUNED (Flan-T5)** | Jeli jest te sowa. (formy biblijne na tekście współczesnym) |

Komentarz: Na OOD baseline częściej zostaje przy EN; finetuned przenosi styl biblijny na tekst współczesny.

---

# Ostatni slajd: Wnioski (3 bullet-pointy)

- **Efekt domenowy:** Fine-tuning na Bible podnosi BLEU/chrF in-domain, obniża wyniki na zbiorach OOD (Contemporary, Technical, Theology).
- **Trade-off:** Większa specjalizacja do domeny treningu = mniejsza generalizacja; wyniki z tabeli to potwierdzają.
- **Cel projektu:** Eksperyment pokazuje dostosowanie modelu do stylu/słownictwa biblijnego; nie rozstrzyga, na ile uczy się ogólnych zasad tłumaczenia EN→PL.
