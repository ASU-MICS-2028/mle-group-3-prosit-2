#!/usr/bin/env python
"""Write a Zindi submission for the African Credit Scoring Challenge from the deployed model.

    python scripts/make_submission.py      -> data/processed/zindi_submission.csv

The deployed logistic regression is fitted on every labelled Train loan and scores the
Test loans; a row is flagged as a default when P(default) is above the cost threshold.
Loan-level scores are copied to each lender row of a co-funded loan. The file stays in
data/ (not committed) because it is derived from the competition data.
"""
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

import pandas as pd  # noqa: E402

from src import config  # noqa: E402
from src import features as F  # noqa: E402
from src import models as M  # noqa: E402
from src.costs import LoanEconomics  # noqa: E402
from src.data import load_loans, to_row_predictions  # noqa: E402


def main() -> int:
    train, test = load_loans("train"), load_loans("test")
    X = F.build_features(pd.concat([train, test], ignore_index=True))
    cols = M.inputs(M.DEPLOY_FEATURES)
    model = M.logreg(numeric=M.DEPLOY_FEATURES, class_weight=None).fit(X.iloc[: len(train)][cols], train.target)
    p = model.predict_proba(X.iloc[len(train):][cols])[:, 1]
    flags = (to_row_predictions(test, p) > LoanEconomics().threshold).astype(int)
    sample = pd.read_csv(config.DATA_RAW / "SampleSubmission.csv")
    out = sample[["ID"]].assign(target=sample.ID.map(flags))
    assert out.target.notna().all(), "some submission rows have no prediction"
    path = config.DATA_PROCESSED / "zindi_submission.csv"
    out.astype({"target": int}).to_csv(path, index=False)
    by_country = out.assign(country=out.ID.map(test.explode("row_ids").set_index("row_ids").country_id)).groupby("country").target.agg(["size", "mean"])
    print(f"Wrote {path.relative_to(ROOT)}: {len(out):,} rows, {int(out.target.sum()):,} flagged ({out.target.mean():.1%})")
    print(by_country.rename(columns={"size": "rows", "mean": "share flagged"}).round(3).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
