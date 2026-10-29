"""The loan-level collapse must keep every loan, every row ID and every default."""
import pytest

from src.data import load_rows, to_loans, to_row_predictions


@pytest.fixture(scope="module")
def train():
    rows = load_rows("train")
    return rows, to_loans(rows)


def test_one_row_per_loan(train):
    rows, loans = train
    assert loans.tbl_loan_id.is_unique
    assert len(loans) == rows.tbl_loan_id.nunique() == 66_520


def test_row_ids_round_trip(train):
    rows, loans = train
    back = to_row_predictions(loans, range(len(loans)))
    assert set(back.index) == set(rows.ID) and len(back) == len(rows)


def test_defaults_and_amounts_preserved(train):
    rows, loans = train
    assert loans.target.sum() == rows.groupby("tbl_loan_id").target.max().sum()
    assert loans.Amount_Funded_By_Lender.sum() == pytest.approx(rows.Amount_Funded_By_Lender.sum())
    assert (loans.n_lenders == 2).sum() == 2_134


def test_test_split_collapses():
    loans = to_loans(load_rows("test"))
    assert "target" not in loans and loans.row_ids.map(len).sum() == 18_594
