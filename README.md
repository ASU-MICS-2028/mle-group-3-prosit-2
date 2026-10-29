# NkwaPay Credit Scoring (ICS553 Prosit 2)

Default prediction and loan sizing for NkwaPay, trained on Kenyan loans and assessed for transfer to Ghana. Debrief: **Tuesday 3 November 2026**.

**Deliverables:** report `reports/NkwaPay_Report_Group3.docx` (source `reports/report.md`) · slides `slides/NkwaPay_Debrief_Group3.pptx` · Model Card `docs/model_card.md` · decline letter `docs/decline_letter.md` · AI-use declaration `docs/ai_use_declaration.md` (draft, to sign) · reflection template `docs/reflection_template.md`.

## Headline results

| | |
|---|---|
| Data | 66,520 Kenyan loans after merging co-funded rows, 1.63% default; 2,084 unlabelled Ghana loans |
| Leakage | `Total_Amount_to_Repay` is updated after default (AUC 0.98); dropped. Naive pipeline PR-AUC 0.825 vs honest 0.145 |
| Validation | Temporal expanding window. Random CV overstates 1.7–2.7× and picks the wrong model |
| Regime change | Type_1 (93% of loans) ends 1 Dec 2022; afterwards defaults 9–13%, ROC ~0.6 |
| Classifier | Logistic regression, decline if P(default) > 0.062 (cedi cost matrix). PR-AUC 0.145, ROC 0.725 (temporal) |
| Loan size | Ridge on log amount (repaid loans), median error ~1,930 KES; first-loan cap needed |
| Ghana | 87% unseen products, 96% unseen lenders, domain AUC 1.0. No automated Ghana decisions before a pilot |
| Service | `Scorer.score()` → P(default), approve/decline/refer, 3 reasons, offer; ~20 ms per call |

## Layout

```
data/raw/         raw files (gitignored, see data/README.md)
data/processed/   features and out-of-fold predictions (gitignored, rebuilt by the notebooks)
docs/             Model Card, decline letter, AI declaration, legal texts (BoG Directive, Act 843)
models/           saved scoring service (gitignored, rebuilt by notebook 12)
notebooks/        one notebook per workplan section, 01 to 12
reports/          report, figures and result tables
scripts/          run_all.sh reruns everything
slides/           debrief deck and its build script
src/              shared code: data, features, models, validation, costs, bootstrap, reasons, scoring
tests/            26 pytest checks (data, leakage replay, loan merge, reasons, scoring service)
```

## Setup and rerun

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # plus pandoc for the report
# put the data in data/raw/ (see data/README.md)
bash scripts/run_all.sh                  # notebooks 01-12 in order, tests, report, slides (~15 min)
```

Notebooks depend on each other's saved outputs: 03 writes `data/processed/features.pkl`, 05 and 06 write out-of-fold predictions used by 07. Run them in order (the script does).

On macOS LightGBM is pinned to one thread (`n_jobs=1`); multithreaded fits were about 40× slower here.

## Conventions

- Notebooks are named `NN_topic.ipynb` to match the workplan sections.
- Shared logic goes in `src/` with a test, not in notebooks.
- Branch per section (`section/03-features`), merge into `main`.
- `random_state=SEED` (42) everywhere; call `set_seed()` at the top of each notebook.

## Progress tracker

| Section | Owner | Status | Notebook / output |
|---|---|---|---|
| 0 Setup | | Done (Act 726 text still not obtained) | repo, `requirements.txt`, `tests/test_data_present.py` |
| 1 Framing (Part A) | | Done | `01_problem_framing.ipynb`, `src/schema.py`, `src/costs.py` |
| 2 EDA (B1) | | Done | `02_eda.ipynb`, `src/data.py` |
| 3 Features (B2) | | Done | `03_features.ipynb`, `src/features.py`, `tests/test_features.py` |
| 4 Validation (B3) | | Done | `04_validation.ipynb`, `src/validation.py` |
| 5 Classification (B4–B5) | | Done | `05_classification.ipynb`, `src/models.py` |
| 6 Regression (B4–B5) | | Done | `06_regression.ipynb` |
| 7 Noise floor (B6) | | Done | `07_noise_floor.ipynb`, `src/bootstrap.py` |
| 8 Derivations (LO3) | | Done | `08_derivations.ipynb` |
| 9 Reason codes (B7) | | Done | `09_reason_codes.ipynb`, `src/reasons.py` |
| 10 Ghana transfer (Part C) | | Done | `10_ghana_transfer.ipynb` |
| 11 Ethics and regulation (Part D) | | Done (Act 726 unverified) | `11_ethics_regulation.ipynb`, `docs/model_card.md`, `docs/decline_letter.md` |
| 12 Scoring service (Part E) | | Done | `12_scoring_service.ipynb`, `src/scoring.py` |
| 13 Report, slides, AI declaration | | Drafted, team review needed | `reports/`, `slides/`, `docs/ai_use_declaration.md` |
| 14 Individual reflection | everyone | To do | `docs/reflection_template.md` |

## Open items for the team

1. Fill in owners above and sign the AI-use declaration.
2. Download Act 726 from the [Parliament repository](https://ir.parliament.gh/handle/123456789/1794) and confirm the bureau-check obligation (reported s.26) in `11_ethics_regulation.ipynb` §11.4.
3. Agree or change the cost-matrix assumptions (`src/costs.py`) and the regression target (repaid loans only).
4. Everyone: rerun `scripts/run_all.sh` and be able to explain each number.
5. Write individual reflections.
