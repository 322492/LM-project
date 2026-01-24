#!/usr/bin/env python3
"""
Generator szablonu dokumentacji dla out-of-domain test sets.

Tworzy:
- data/ood/README.md z checklistą do uzupełnienia
- opcjonalnie: DUMMY placeholdery z losowej próbki testu biblijnego
  (tylko do testowania pipeline, wyraźnie oznaczone jako DUMMY)
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import List

from config_utils import get_nested, load_toml, pick


def read_lines(path: Path) -> List[str]:
    with path.open("r", encoding="utf-8", errors="replace") as f:
        return [line.rstrip("\n") for line in f]


def write_lines(path: Path, lines: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", errors="replace") as f:
        for line in lines:
            f.write(line + "\n")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def generate_readme(ood_dir: Path) -> None:
    """Generuje README.md z checklistą dla OOD test sets."""
    readme_content = """# Out-of-domain test sets (EN→PL)

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
"""
    write_text(ood_dir / "README.md", readme_content)
    print(f"[OK] Utworzono: {ood_dir / 'README.md'}")


def generate_dummy_placeholders(
    ood_dir: Path, test_en_path: Path, test_pl_path: Path, n_contemporary: int = 300, n_technical: int = 300, seed: int = 42
) -> None:
    """
    Generuje DUMMY placeholdery z losowej próbki testu biblijnego.
    
    UWAGA: To są DUMMY dane — nie testują generalizacji, tylko pipeline.
    """
    print("[WARNING] GENEROWANIE DUMMY PLACEHOLDERÓW (z testu biblijnego)")
    print("   Te dane NIE testują generalizacji — służą tylko do testowania pipeline!")
    print()

    test_en = read_lines(test_en_path)
    test_pl = read_lines(test_pl_path)

    if len(test_en) != len(test_pl):
        print(f"ERROR: różna liczba linii w test.en ({len(test_en)}) i test.pl ({len(test_pl)})")
        return

    n_total = len(test_en)
    n_needed = n_contemporary + n_technical
    if n_needed > n_total:
        print(f"WARNING: potrzebujesz {n_needed} par, ale test ma tylko {n_total}. Używam wszystkich.")
        n_contemporary = min(n_contemporary, n_total // 2)
        n_technical = n_total - n_contemporary

    rng = random.Random(seed)
    indices = list(range(n_total))
    rng.shuffle(indices)

    # Contemporary
    cont_indices = sorted(indices[:n_contemporary])
    cont_en = [test_en[i] for i in cont_indices]
    cont_pl = [test_pl[i] for i in cont_indices]

    # Technical
    tech_indices = sorted(indices[n_contemporary : n_contemporary + n_technical])
    tech_en = [test_en[i] for i in tech_indices]
    tech_pl = [test_pl[i] for i in tech_indices]

    # Zapisz z DUMMY oznaczeniem
    dummy_header = [
        "# DUMMY PLACEHOLDER - NIE UŻYWAJ DO EWALUACJI GENERALIZACJI!",
        "# Wygenerowano z losowej próbki testu biblijnego (tylko do testowania pipeline).",
        "#",
    ]

    write_lines(ood_dir / "contemporary.en", dummy_header + cont_en)
    write_lines(ood_dir / "contemporary.pl", dummy_header + cont_pl)
    write_lines(ood_dir / "technical.en", dummy_header + tech_en)
    write_lines(ood_dir / "technical.pl", dummy_header + tech_pl)

    print(f"[OK] Utworzono DUMMY placeholdery:")
    print(f"  - contemporary: {n_contemporary} par")
    print(f"  - technical: {n_technical} par")
    print(f"  [WARNING] Pamiętaj: to są DUMMY dane z testu biblijnego!")


def main() -> int:
    ap = argparse.ArgumentParser(description="Generator szablonu dokumentacji dla OOD test sets.")
    ap.add_argument("--config", type=Path, default=Path("configs/default.toml"), help="Plik config TOML.")
    ap.add_argument(
        "--generate-dummy",
        action="store_true",
        help="Generuj DUMMY placeholdery z losowej próbki testu biblijnego (tylko do testowania pipeline).",
    )
    ap.add_argument("--ood-dir", type=Path, default=None, help="Katalog OOD. (nadpisuje config)")
    ap.add_argument("--test-en", type=Path, default=None, help="Plik test.en. (nadpisuje config)")
    ap.add_argument("--test-pl", type=Path, default=None, help="Plik test.pl. (nadpisuje config)")
    ap.add_argument(
        "--n-contemporary",
        type=int,
        default=300,
        help="Liczba par dla contemporary (DUMMY). (domyślnie: 300)",
    )
    ap.add_argument("--n-technical", type=int, default=300, help="Liczba par dla technical (DUMMY). (domyślnie: 300)")
    ap.add_argument("--seed", type=int, default=42, help="Seed dla losowania DUMMY. (domyślnie: 42)")
    args = ap.parse_args()

    cfg = load_toml(Path(args.config))

    ood_dir = Path(pick(args.ood_dir, get_nested(cfg, ["paths", "ood_dir"]), "data/ood"))
    test_en_path = Path(
        pick(args.test_en, get_nested(cfg, ["paths", "splits_random_test_en"]), "data/splits_random/test.en")
    )
    test_pl_path = Path(
        pick(args.test_pl, get_nested(cfg, ["paths", "splits_random_test_pl"]), "data/splits_random/test.pl")
    )

    # Zawsze generuj README.md
    generate_readme(ood_dir)

    # Opcjonalnie: generuj DUMMY placeholdery
    if args.generate_dummy:
        if not test_en_path.exists() or not test_pl_path.exists():
            print(f"ERROR: nie znaleziono plików testowych:")
            print(f"  - {test_en_path}")
            print(f"  - {test_pl_path}")
            return 1
        generate_dummy_placeholders(ood_dir, test_en_path, test_pl_path, args.n_contemporary, args.n_technical, args.seed)
    else:
        print()
        print("Aby wygenerować DUMMY placeholdery (tylko do testowania pipeline):")
        print(f"  python scripts/ood_template_builder.py --generate-dummy")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
