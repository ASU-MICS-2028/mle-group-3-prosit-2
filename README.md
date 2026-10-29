# NkwaPay Credit Scoring (ICS553 Prosit 2)

Default prediction and loan sizing for NkwaPay, trained on Kenyan loans and assessed for transfer to Ghana. Debrief: **Tuesday 3 November 2026**.

## Layout

```
data/raw/         raw files (gitignored, see data/README.md)
data/processed/   intermediate outputs (gitignored)
docs/legal/       BoG Directive, Act 843, Act 726
notebooks/        one notebook per workplan section, prefixed by number
src/              shared code (config, features, models, scoring service)
tests/            pytest checks (data shapes, leakage)
reports/          written report and figures
slides/           debrief deck
```

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# put the data in data/raw/ (see data/README.md)
pytest tests/
```

Start each notebook with:

```python
import sys; sys.path.append("..")
from src.config import SEED, set_seed, TRAIN_CSV
set_seed()
```

and pass `random_state=SEED` to every split and model.

## Conventions

- Notebooks are named `NN_topic.ipynb` to match the workplan sections (e.g. `02_eda.ipynb`).
- Clear outputs before committing large notebooks.
- Branch per section (`section/03-features`), PR into `main`.

## Owners

| Section | Owner | Due |
|---|---|---|
| 0 Setup | | Wed 28 Oct |
| 1 Framing (Part A) | | Thu 29 Oct |
| 2 EDA (B1) | | Thu 29 Oct |
| 3 Features (B2) | | Fri 30 Oct |
| 4 Validation (B3) | | Fri 30 Oct |
| 5 Classification (B4–B5) | | Sat 31 Oct |
| 6 Regression (B4–B5) | | Sat 31 Oct |
| 7 Noise floor (B6) | | Sun 1 Nov |
| 8 Derivations (LO3) | | Sun 1 Nov |
| 9 Reason codes (B7) | | Sun 1 Nov |
| 10 Ghana transfer (Part C) | | Sun 1 Nov |
| 11 Ethics and regulation (Part D) | | Mon 2 Nov |
| 12 Scoring service (Part E) | | Mon 2 Nov |
| 13 Report, slides, AI declaration | | Mon 2 Nov |
| 14 Individual reflection | everyone | Mon 2 Nov |
