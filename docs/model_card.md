# Model Card: NkwaPay default-risk classifier (v1, ICS553 Prosit 2)

## Model details
- **Type:** logistic regression (scikit-learn, L2, C = 1, unweighted), with median imputation plus missing-value indicators, standardisation, and one-hot product and lender (levels with fewer than 30 training loans grouped as Other, unseen levels as Unknown).
- **Inputs (deployed set):** loan terms (log amount, term, amount per day, lender share, number of co-funders, New/Repeat flag), borrower history from earlier loans only (counts, observable past defaults, active loans, past amounts and terms, distinct lenders), product and lender. Calendar and macro features excluded. Full list: `src/models.py::DEPLOY_FEATURES`.
- **Output:** probability of default; decision rule decline if P(default) > 0.062 (from the cedi cost matrix, `src/costs.py`); top three reason codes (`src/reasons.py`).
- **Companion model:** ridge regression for a loan-size limit on repaid loans (Section 6).
- **Owners:** ICS553 group 3. Date: November 2026.

## Intended use
- Ranking and screening **Kenyan** digital loan applications similar to those in the training data, with human review of declines on request.
- **Not intended for:** automated decisions in Ghana or any market without local outcome data; decisions about people with no relationship to the products and lenders seen in training; any use other than credit assessment.

## Training data
- Zindi African Credit Scoring Challenge, Kenya, Oct 2021 to Nov 2024: 66,520 loans after merging co-funded rows, 1.63% default.
- Excluded as leakage: Total_Amount_to_Repay and Lender_portion_to_be_repaid (updated after default).
- No protected or special personal data (Act 843 s.37).

## Evaluation (temporal expanding-window CV, 5 folds)
- PR-AUC 0.145, ROC-AUC 0.725 (fold average). Bootstrap 95% interval for PR-AUC about ±0.025 (Section 7).
- 2022 boom regime: ROC about 0.78. From Dec 2022: ROC about 0.64, PR-AUC lift about 1.8 over base rate.
- Beats the approve-everyone baseline on every metric and in cedis; not distinguishable from shallow LightGBM on PR-AUC overall.
- Calibration: good in the 2022 regime; over-spread after Dec 2022 (low scores too low, high scores too high).

## Segment performance (Section 11.2)
- Largest gaps by borrowing history and New vs Repeat: thin-file borrowers are declined far more often, including many who would repay.
- Decline rates differ by lender and product by an order of magnitude.

## Ethical considerations
- Borrowing history is a likely proxy for gender, income, education, age and rural residence in Ghana (Findex 2021, Section 11.5).
- Mitigation: no automatic decline for thin-file applicants; capped starter loans; monthly parity monitoring by history band; consented demographic sample for audits.
- Reason codes are relative to the average applicant, not counterfactuals; product and lender reasons are not actionable by the borrower.

## Caveats and recommendations
- **Ghana:** 87% of Ghana loans are on unseen products and 96% on unseen lenders; a domain classifier separates Ghana from Kenya with AUC 1.0. Do not use for automated Ghana decisions until pilot outcomes allow recalibration and validation (Section 10).
- **Drift:** performance roughly halved after Dec 2022. Recalibrate monthly on recent outcomes and monitor approval rate, default rate and mean prediction.
- **Regulation:** no credit-bureau integration (BoG Directive §7.3; Act 726 reportedly requires a credit report). Act 726 and Act 1052 not reviewed from primary text.
- Cost-matrix figures are assumptions; replace with NkwaPay's own.
