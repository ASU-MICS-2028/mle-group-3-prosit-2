"""Load the Zindi files and collapse them to one row per loan (Section 2 decision).

A loan co-funded by two lenders appears on two rows that share every loan-level
field. Modelling at row level would double-count those loans (which default far
more often) and let one half of a loan sit in train while the other half is
validated. We model at loan level and broadcast predictions back to row IDs.
"""
import pandas as pd

from src import config

DATES = ["disbursement_date", "due_date"]
LOAN_FIELDS = ["customer_id", "country_id", "loan_type", "Total_Amount", "Total_Amount_to_Repay",
               "disbursement_date", "due_date", "duration"]


def load_rows(split: str = "train") -> pd.DataFrame:
    path = {"train": config.TRAIN_CSV, "test": config.TEST_CSV}[split]
    return pd.read_csv(path, parse_dates=DATES)


def to_loans(rows: pd.DataFrame) -> pd.DataFrame:
    """One row per tbl_loan_id.

    - Loan-level fields are identical across a loan's rows, so take the first.
    - Lender shares and amounts are summed; n_lenders counts co-funders.
    - lender_id is the lender with the largest share (ties: lowest id).
    - New_versus_Repeat is 'Repeat Loan' if any lender row says so: the 133
      conflicting loans are new to one lender but not to the platform.
    - target is 1 if any row defaulted (2 loans conflict in Train).
    """
    rows = rows.sort_values(["tbl_loan_id", "Lender_portion_Funded", "lender_id"], ascending=[True, False, True])
    g = rows.groupby("tbl_loan_id", sort=False)
    agg = {c: "first" for c in LOAN_FIELDS}
    agg.update(lender_id="first", Lender_portion_Funded="sum", Amount_Funded_By_Lender="sum",
               Lender_portion_to_be_repaid="sum")
    loans = g.agg(agg)
    loans["n_lenders"] = g.size()
    loans["lenders"] = g.lender_id.agg(lambda s: "+".join(map(str, sorted(s))))
    loans["New_versus_Repeat"] = g.New_versus_Repeat.agg(lambda s: "Repeat Loan" if (s == "Repeat Loan").any() else "New Loan")
    loans["row_ids"] = g.ID.agg(list)
    if "target" in rows:
        loans["target"] = g.target.max()
    return loans.reset_index().sort_values(["disbursement_date", "tbl_loan_id"], ignore_index=True)


def load_loans(split: str = "train") -> pd.DataFrame:
    return to_loans(load_rows(split))


def to_row_predictions(loans: pd.DataFrame, scores) -> pd.Series:
    """Broadcast loan-level scores back to the original row IDs (e.g. for a Zindi submission)."""
    s = loans[["row_ids"]].assign(score=list(scores)).explode("row_ids")
    return s.set_index("row_ids").score.rename_axis("ID")
