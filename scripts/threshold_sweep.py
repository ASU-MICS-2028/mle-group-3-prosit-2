#!/usr/bin/env python
"""Precision, recall and F1 of the deployed model at swept thresholds (brief, Part B5).

    python scripts/threshold_sweep.py      -> reports/13_threshold_sweep.csv

Needs no data. Notebook 13 already saved, per threshold, the approval rate and the share
of defaults let through (reports/13_approval_goal.csv), and the loan and default counts per
scope (reports/13_confusion_matrix.csv). The confusion counts follow from those exactly:
  TP = defaults x (1 - let through),  declined = loans x (1 - approval rate),  FP = declined - TP.
The script checks that it rebuilds the saved confusion matrix at the cost threshold.
"""
from pathlib import Path

import pandas as pd

REPORTS = Path(__file__).resolve().parents[1] / "reports"


def main() -> None:
    goal = pd.read_csv(REPORTS / "13_approval_goal.csv")
    cm = pd.read_csv(REPORTS / "13_confusion_matrix.csv", index_col=0)
    loans = goal.scope.map(cm.loans)
    defaults = goal.scope.map(cm["TP (defaulter declined)"] + cm["FN (defaulter approved)"])
    tp = (defaults * (1 - goal["defaults let through"])).round()
    declined = (loans * (1 - goal["approval rate"])).round()
    fp = declined - tp
    out = pd.DataFrame({
        "threshold": goal.threshold, "scope": goal.scope, "loans": loans, "defaults": defaults,
        "TP": tp, "FP": fp, "FN": defaults - tp, "TN": loans - defaults - fp,
        "precision": (tp / declined).where(declined > 0), "recall": tp / defaults,
        "F1": (2 * tp / (2 * tp + fp + defaults - tp)), "approval rate": goal["approval rate"],
        "GHS per 1,000": goal["GHS per 1,000"],
    })
    # Must reproduce the saved confusion matrix at the cost threshold (0.0617), scope by scope
    at = out[out.threshold.round(4) == 0.0617].set_index("scope")
    for col, ref in [("TP", "TP (defaulter declined)"), ("FN", "FN (defaulter approved)"),
                     ("FP", "FP (good borrower declined)"), ("TN", "TN (good borrower approved)")]:
        assert (at[col] == cm.loc[at.index, ref]).all(), col
    assert (at.F1 - cm.loc[at.index, "F1"]).abs().max() < 1e-9
    out.to_csv(REPORTS / "13_threshold_sweep.csv", index=False)
    print(out[out.scope == "all folds"].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
