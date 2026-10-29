"""Column roles from the Part A feature audit (notebooks/01_problem_framing.ipynb).

Decision point: the moment NkwaPay approves or declines an application, after the
borrower has chosen amount, product and term but before money is disbursed.
Anything not known at that moment must not be a feature.
"""

TARGET = "target"  # 1 = defaulted, 0 = repaid

# Identifiers: never model inputs. Used for grouping, joins and borrower history.
IDS = ["ID", "customer_id", "tbl_loan_id"]
GROUP_KEY = "customer_id"
LOAN_KEY = "tbl_loan_id"

# Known at the decision point (application terms and product choice).
USABLE = [
    "loan_type",
    "lender_id",
    "Total_Amount",
    "duration",
    "New_versus_Repeat",
    "Lender_portion_Funded",
    "disbursement_date",  # time features and validation splits; ~ application date
]

# Usable in the classifier, but not in the loan-size regression (they are the
# amount itself or scale directly with it).
AMOUNT_DERIVED = ["Total_Amount", "Amount_Funded_By_Lender"]

# Excluded, with the reason recorded in the Part A audit table.
DROPPED = {
    "country_id": "Constant (Kenya) in Train; kept only for Ghana split and macro join",
    "due_date": "Redundant: equals disbursement_date + duration on every row",
    "Amount_Funded_By_Lender": "Redundant: equals Lender_portion_Funded x Total_Amount",
    "Total_Amount_to_Repay": "Suspected post-outcome leakage: inflated for defaulters on identical contracts (AUC of repay/amount = 0.98)",
    "Lender_portion_to_be_repaid": "Derived from Total_Amount_to_Repay, inherits the same leakage",
}

# Kept for metadata and joins, not as a direct model input.
META = ["country_id"]

LEAKY = ["Total_Amount_to_Repay", "Lender_portion_to_be_repaid"]
