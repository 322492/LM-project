# Out-of-domain test sets (EN→PL)

Ten katalog zawiera **out-of-domain** test sets do ewaluacji generalizacji modelu.

## Zbiory testowe

### 1. Contemporary (teksty współczesne)
- Pliki: `contemporary.en`, `contemporary.pl`
- **Status**: ⚠️ PLACEHOLDER (wymaga uzupełnienia)

**Checklista do uzupełnienia:**
- [ ] Źródło danych: _______________________
- [ ] Link do źródła: _______________________
- [ ] Licencja: _______________________
- [ ] Kryteria doboru zdań: _______________________
- [ ] Liczba par: _______________________
- [ ] Jak zapewniamy brak przecieku do treningu (Biblia): _______________________

**Docelowo:**
- 200-500 par zdań EN-PL
- Teksty z lat 2000+ (współczesne)
- Różne domeny: news, literatura, blogi, fora
- Brak przecieku do danych treningowych

---

### 2. Technical (teksty techniczne)
- Pliki: `technical.en`, `technical.pl`
- **Status**: ⚠️ PLACEHOLDER (wymaga uzupełnienia)

**Checklista do uzupełnienia:**
- [ ] Źródło danych: _______________________
- [ ] Link do źródła: _______________________
- [ ] Licencja: _______________________
- [ ] Kryteria doboru zdań: _______________________
- [ ] Liczba par: _______________________
- [ ] Jak zapewniamy brak przecieku do treningu (Biblia): _______________________

**Docelowo:**
- 200-500 par zdań EN-PL
- Teksty techniczne: IT, nauka, instrukcje, dokumentacja
- Terminologia specjalistyczna
- Brak przecieku do danych treningowych

---

## Uwagi

- Pliki są w formacie **Moses** (1 linia = 1 para zdań).
- Jeśli używasz **DUMMY placeholderów** (wygenerowanych z testu biblijnego), są one wyraźnie oznaczone w plikach.
- **DUMMY placeholdery służą tylko do testowania pipeline** — nie używaj ich do prawdziwej ewaluacji generalizacji.

## Generowanie DUMMY placeholderów

Aby wygenerować DUMMY placeholdery z losowej próbki testu biblijnego (tylko do testowania):

```bash
python scripts/ood_template_builder.py --generate-dummy
```

**UWAGA:** To są DUMMY dane z testu biblijnego, więc **NIE testują generalizacji** — służą tylko do sprawdzenia, że pipeline działa.
