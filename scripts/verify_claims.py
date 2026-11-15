#!/usr/bin/env python
"""Recompute the headline numbers quoted in the report, README and slides.

    python scripts/verify_claims.py          (from the repo root, data in data/raw/)

Each check recomputes one number from the raw files and compares it with the
value we quote. The counts and shares are computed here with plain pandas,
not by calling the notebooks, so a mistake in a notebook does not hide itself.
The cross-validation checks refit the deployed model on the temporal folds
(about ten seconds). Only aggregates are printed, never loan rows.

Exit code 0 if every check passes, 1 otherwise. The claim IDs match
reports/claims_table.md.
"""
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

from src import config  # noqa: E402
from src.costs import LoanEconomics  # noqa: E402

RESULTS = []


def check(cid: str, what: str, quoted: float, got: float, tol: float) -> None:
    """Record one claim. `tol` is the largest absolute gap that still counts as a match."""
    ok = bool(np.isfinite(got) and abs(got - quoted) <= tol)
    RESULTS.append((cid, what, quoted, got, ok))


def raw_checks():
    dates = ["disbursement_date", "due_date"]
    tr = pd.read_csv(config.TRAIN_CSV, parse_dates=dates)
    te = pd.read_csv(config.TEST_CSV, parse_dates=dates)

    check("C01", "Train rows", 68_654, len(tr), 0)
    check("C02", "Train rows that defaulted", 1_258, int(tr.target.sum()), 0)
    check("C03", "Default rate per row (%)", 1.83, 100 * tr.target.mean(), 0.005)

    # One row per loan: a co-funded loan is in default if either lender row says so.
    per_loan = tr.groupby("tbl_loan_id").agg(target=("target", "max"), n_lenders=("ID", "size"))
    check("C04", "Train loans after merging co-funded rows", 66_520, len(per_loan), 0)
    check("C05", "Default rate per loan (%)", 1.63, 100 * per_loan.target.mean(), 0.005)
    co = per_loan.n_lenders > 1
    check("C06", "Co-funded loans in Train", 2_134, int(co.sum()), 0)
    check("C07", "Default rate, co-funded loans (%)", 8.3, 100 * per_loan.target[co].mean(), 0.05)
    check("C08", "Default rate, single-lender loans (%)", 1.4, 100 * per_loan.target[~co].mean(), 0.05)

    check("C09", "Loans with one row in Train and one in Test", 920,
          len(set(tr.tbl_loan_id) & set(te.tbl_loan_id)), 0)
    te_cust = te.drop_duplicates("customer_id")
    seen = te_cust.customer_id.isin(tr.customer_id)
    check("C10", "Test customers who also appear in Train", 4_306, int(seen.sum()), 0)
    check("C11", "Share of Test customers also in Train (%)", 87, 100 * seen.mean(), 0.5)
    check("C12", "Ghana Test customers seen in Train", 0, int(seen[te_cust.country_id == "Ghana"].sum()), 0)

    ratio = tr.Total_Amount_to_Repay / tr.Total_Amount
    check("C13", "AUC of repay/amount ratio against default", 0.98, roc_auc_score(tr.target, ratio), 0.005)
    same = (tr.loan_type == "Type_1") & (tr.duration == 7) & (tr.lender_id == 267278)
    med = ratio[same].groupby(tr.target[same]).median()
    check("C14", "Median owed over principal, repaid, same contract (%)", 0.7, 100 * (med[0] - 1), 0.05)
    check("C15", "Median owed over principal, defaulted, same contract (%)", 41, 100 * (med[1] - 1), 0.5)
    gap = (tr.due_date - tr.disbursement_date).dt.days - tr.duration
    check("C16", "Rows where due date differs from disbursement + term", 0, int((gap != 0).sum()), 0)

    loans = tr.drop_duplicates("tbl_loan_id")
    check("C17", "Share of loans with a 7-day term (%)", 96, 100 * (loans.duration == 7).mean(), 0.5)
    types = loans.loan_type.value_counts()
    check("C18", "Loan types in Train", 22, len(types), 0)
    check("C19", "Type_1 share of loans (%)", 93, 100 * types.iloc[0] / len(loans), 0.5)
    check("C20", "Loan types with fewer than 30 loans", 11, int((types < 30).sum()), 0)

    # Ghana, at loan level, using each loan's largest lender as in src/data.py.
    from src.data import load_loans
    ke, te_loans = load_loans("train"), load_loans("test")
    gh = te_loans[te_loans.country_id == "Ghana"]
    check("C21", "Ghana Test loans", 2_084, len(gh), 0)
    check("C22", "Ghana loans on a product never seen in Kenya (%)", 87, 100 * (~gh.loan_type.isin(ke.loan_type)).mean(), 0.5)
    check("C23", "Ghana loans with a lender never seen in Kenya (%)", 96, 100 * (~gh.lender_id.isin(ke.lender_id)).mean(), 0.5)
    check("C24", "Ghana loans that are co-funded (%)", 67, 100 * (gh.n_lenders > 1).mean(), 0.5)


def cost_checks():
    e = LoanEconomics()
    check("C25", "Gain from a repaid GHS 500 loan (GHS)", 25, e.gain_good, 0.005)
    check("C26", "Loss from a defaulted GHS 500 loan (GHS)", 380, e.loss_bad, 0.005)
    check("C27", "Decline threshold on P(default)", 0.062, e.threshold, 0.0005)
    from dataclasses import replace
    check("C28", "Threshold with no recovery (%)", 4.5, 100 * replace(e, recovery_rate=0.0).threshold, 0.05)
    check("C29", "Threshold with 80% recovery (%)", 16, 100 * replace(e, recovery_rate=0.8).threshold, 0.5)


def model_checks():
    """Refit the deployed classifier on the temporal folds and recount the confusion matrix."""
    from src import features as F
    from src import models as M
    from src import validation as V
    from src.data import load_loans

    cache = config.DATA_PROCESSED / "features.pkl"
    if cache.exists():
        feat = pd.read_pickle(cache)
        tr = feat[feat.split == "train"].reset_index(drop=True)
    else:  # build the features from the raw files, as notebook 03 does
        raw_tr, raw_te = load_loans("train"), load_loans("test")
        tr = F.build_features(pd.concat([raw_tr, raw_te], ignore_index=True)).iloc[: len(raw_tr)].reset_index(drop=True)
    cols = M.inputs(M.DEPLOY_FEATURES)
    oof, fold, folds = V.cross_validate(M.logreg(numeric=M.DEPLOY_FEATURES, class_weight=None), tr, cols, V.temporal(tr))
    check("C30", "PR-AUC, deployed model, temporal folds", 0.145, folds.pr_auc.mean(), 0.0005)
    check("C31", "ROC-AUC, deployed model, temporal folds", 0.725, folds.roc_auc.mean(), 0.0005)

    tested = fold >= 0
    y = tr.target.to_numpy().astype(int)[tested]
    declined = oof[tested] > LoanEconomics().threshold
    late = fold[tested] >= 3
    tp, fn = int((declined & (y == 1)).sum()), int((~declined & (y == 1)).sum())
    fp, tn = int((declined & (y == 0)).sum()), int((~declined & (y == 0)).sum())
    check("C32", "Loans scored out of time", 39_894, int(tested.sum()), 0)
    check("C33", "Defaulters declined (TP)", 188, tp, 0)
    check("C34", "Defaulters approved (FN)", 467, fn, 0)
    check("C35", "Good borrowers declined (FP)", 1_135, fp, 0)
    check("C36", "Good borrowers approved (TN)", 38_104, tn, 0)
    check("C37", "Precision at the threshold", 0.14, tp / (tp + fp), 0.005)
    check("C38", "Recall at the threshold", 0.29, tp / (tp + fn), 0.005)
    check("C39", "F1 at the threshold", 0.19, 2 * tp / (2 * tp + fp + fn), 0.005)
    check("C40", "Approval rate, all folds (%)", 96.7, 100 * (1 - declined.mean()), 0.05)
    check("C41", "Approval rate, 2022 regime (%)", 98.8, 100 * (1 - declined[~late].mean()), 0.05)
    check("C42", "Approval rate, from Dec 2022 (%)", 68.6, 100 * (1 - declined[late].mean()), 0.05)

    thin = (tr.n_prev_resolved == 0).to_numpy()[tested]
    referred = thin & declined
    check("C51", "Good borrowers with no repayment record above the threshold (%)", 22, 100 * declined[thin & (y == 0)].mean(), 0.5)
    check("C52", "Thin-file applicants above the threshold", 665, int(referred.sum()), 0)
    check("C53", "Of those, repaid", 539, int((referred & (y == 0)).sum()), 0)
    had = tr.n_prev_defaults >= 1
    check("C54", "Train loans made after an earlier default was on record", 89, int(had.sum()), 0)
    check("C55", "Default rate of those loans (%)", 18, 100 * tr.target[had].mean(), 0.5)
    past = (tr.n_prev_defaults >= 1).to_numpy()[tested]
    check("C56", "Such applicants approved by the model alone, out of time", 29, int((past & ~declined).sum()), 0)

    # Fees actually paid on repaid loans, and the threshold they imply for a 7-day loan (Section 13.7).
    raw = pd.read_csv(config.TRAIN_CSV).drop_duplicates("tbl_loan_id").set_index("tbl_loan_id")
    fee = tr.tbl_loan_id.map(raw.Total_Amount_to_Repay) / tr.Total_Amount - 1
    week = (tr.duration <= 7) & (tr.target == 0)
    check("C57", "Mean fee on repaid loans of 7 days or less (%)", 1.4, 100 * fee[week].mean(), 0.05)
    e = LoanEconomics()
    fp_cost = fee[week].mean() * tr.Total_Amount[week].median()
    fn_cost = tr.Total_Amount[week].median() * (1 - e.recovery_rate) + e.collection_cost * 12.7
    check("C58", "Break-even threshold for a 7-day loan at observed fees (%)", 1.9, 100 * fp_cost / (fp_cost + fn_cost), 0.05)

    # The loan limit, out of time (Section 13.8).
    reg_cols = M.inputs(M.REG_FEATURES)
    offer, asked, outcome = [], [], []
    for a, b in V.temporal(tr):
        rep = tr.iloc[a][tr.iloc[a].target == 0]
        lim = np.expm1(M.ridge().fit(rep[reg_cols], np.log1p(rep.Total_Amount)).predict(tr.iloc[b][reg_cols]))
        t = tr.iloc[b]
        o = np.minimum(t.Total_Amount.to_numpy(), lim)
        o = np.where(t.n_prev_resolved.to_numpy() == 0, np.minimum(o, rep[rep.n_prev_loans == 0].Total_Amount.median()), o)
        offer.append(o); asked.append(t.Total_Amount.to_numpy()); outcome.append(t.target.to_numpy())
    offer, asked, outcome = map(np.concatenate, (offer, asked, outcome))
    bad = outcome == 1
    check("C59", "Share of money lent that went to defaulters, no limit (%)", 18, 100 * asked[bad].sum() / asked.sum(), 0.5)
    check("C60", "Same share with the ridge limit (%)", 4.6, 100 * offer[bad].sum() / offer.sum(), 0.05)
    check("C61", "Cut in money reaching eventual defaulters (%)", 87, 100 * (1 - offer[bad].sum() / asked[bad].sum()), 0.5)
    check("C62", "Good borrowers offered their full request (%)", 43, 100 * (offer[~bad] >= asked[~bad] * 0.999).mean(), 0.5)

    saved = config.MODELS / "nkwapay-v1.joblib"
    if saved.exists():
        from src.scoring import Scorer
        s = Scorer.load(saved)
        check("C43", "Shipped model: decline threshold", 0.062, s.threshold, 0.0005)
        check("C44", "Shipped model: loans it was trained on", 66_520, s.meta["n_train"], 0)
        p_saved = s.classifier.predict_proba(tr[cols])[:, 1]
        p_fresh = M.logreg(numeric=M.DEPLOY_FEATURES, class_weight=None).fit(tr[cols], tr.target).predict_proba(tr[cols])[:, 1]
        check("C45", "Shipped model vs a fresh refit: largest gap in P(default)", 0, float(np.abs(p_saved - p_fresh).max()), 1e-6)


def findex_checks():
    fx = pd.read_csv(config.FINDEX_CSV, encoding_errors="replace")
    share = lambda col, d=fx: 100 * np.average(d[col], weights=d.wgt)  # noqa: E731
    fx["borrowed_formal"] = (fx.fin22a == 1).astype(int)
    check("C46", "Ghana adults with a mobile money account (%)", 60, share("account_mob"), 0.5)
    check("C47", "Ghana adults who borrowed from a formal institution (%)", 7, share("borrowed_formal"), 0.5)
    check("C48", "Ghana adults who borrowed any money (%)", 52, share("borrowed"), 0.5)
    check("C49", "Account ownership, women (%)", 63, share("account", fx[fx.female == 1]), 0.5)
    check("C50", "Account ownership, men (%)", 74, share("account", fx[fx.female == 2]), 0.5)


def main() -> int:
    missing = [p.name for p in (config.TRAIN_CSV, config.TEST_CSV, config.FINDEX_CSV) if not p.exists()]
    if missing:
        print(f"Missing in data/raw/: {', '.join(missing)}. See data/README.md.")
        return 2
    for step in (raw_checks, cost_checks, model_checks, findex_checks):
        step()
    width = max(len(w) for _, w, *_ in RESULTS)
    print(f"{'id':<4} {'claim':<{width}} {'quoted':>10} {'recomputed':>12}")
    for cid, what, quoted, got, ok in RESULTS:
        print(f"{cid:<4} {what:<{width}} {quoted:>10,.4g} {got:>12,.4f}  {'ok' if ok else 'MISMATCH'}")
    bad = [r for r in RESULTS if not r[-1]]
    print(f"\n{len(RESULTS) - len(bad)} of {len(RESULTS)} claims reproduce." + ("" if not bad else " See the rows marked MISMATCH."))
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
