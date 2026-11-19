---
title: "NkwaPay Credit Scoring: Kenya Model, Ghana Launch"
subtitle: "ICS553 Prosit 2, Group 3"
author:
  - Mubarak Tijani
  - Eric Elikplim Sunu
  - Clifford Yeboah
  - Adzeyena Opei
date: "3 November 2026"
header-includes:
  - \usepackage{float}
  - \floatplacement{figure}{H}
---

# 1. Summary

NkwaPay wants to score digital loan applications in Ghana using a model trained on Kenyan loans. We built two models from 66,520 labelled Kenyan loans: a classifier that estimates the probability of default, and a regression that suggests how much to lend. We then asked whether either can be trusted in Ghana.

The main findings:

- **The raw data invites a misleading model.** `Total_Amount_to_Repay` is updated after a loan defaults (penalties), so it predicts default almost perfectly (AUC 0.98). A naive pipeline that uses it scores PR-AUC 0.825; our honest out-of-time estimate is 0.145.
- **Validation design changes the answer.** Random cross-validation overstates performance and picks the wrong model: LightGBM wins under stratified CV but loses to logistic regression on future loans.
- **The Kenyan market changed in December 2022.** Type_1, the dominant low-risk product (93% of loans), stops after 1 December 2022. Afterwards defaults run at 9 to 13% and both models fall to ROC about 0.6.
- **The chosen model** is unweighted logistic regression with a decline threshold of P(default) > 0.062, derived from a cedi cost matrix. It beats approving everyone on every metric; it cannot be separated from shallow LightGBM on PR-AUC overall, but it ranks significantly better after December 2022 and gives exact reason codes.
- **Ghana is out of reach for now.** 87% of Ghana loans are on products never seen in Kenya and 96% on unseen lenders; a classifier separates Ghana from Kenya with AUC 1.0. We recommend no automated Ghana decisions until pilot outcomes exist.

All results are reproducible from the repository (Section 13).

# 2. Problem framing (Part A)

**Decision point.** The model runs at approval: the applicant has chosen product, amount and term, and nothing has been disbursed (Figure 1). Any value set after that moment cannot be a feature. The data has no application or approval timestamp, so the closest column is `disbursement_date`. We assume an instant mobile-money loan is approved and paid out on the same day; the data cannot confirm it.

![Loan timeline and decision point](figures/01_loan_timeline.png){width=95%}

**Label.** `target = 1` means the loan defaulted. It is measured over the loan's own term, from `disbursement_date` to `due_date` (disbursement plus `duration`). The data does not state the default rule (days past due or grace period); we treat outcomes as known on the due date (a lag of 0 days, `LABEL_LAG_DAYS` in `src/features.py`) and flag this for NkwaPay.

**Feature audit** (`reports/01_feature_audit.csv`). Every column was classified as usable, identifier, redundant or leaky, with whether it is available at the decision and which task it is the target of. Three columns are not available at the decision: `Total_Amount_to_Repay`, `Lender_portion_to_be_repaid` and `target`. For `Lender_portion_Funded`, and `Amount_Funded_By_Lender` built from it, availability is assumed until NkwaPay confirms the funding split is fixed before approval. The Task 2 target is `target`; the Task 1 target is `Total_Amount` (as `log1p`, on repaid loans). Two decisions needed evidence. `due_date` equals `disbursement_date + duration` on every row, and `Amount_Funded_By_Lender` equals share times amount, so both are redundant. `Total_Amount_to_Repay` looks like a contract term but is not: on the same product, term and lender, repaid loans owe about 0.7% over principal and defaulted loans about 41% (Figure 2). We dropped it and the column derived from it.

![Repay ratio on identical contracts: defaulters owe far more](figures/01_repay_ratio_leakage.png){width=75%}

**Two tasks.** Task 2 (classification) predicts default and is scored with PR-AUC, because only 1.63% of loans default. Task 1 (regression) predicts `log(1 + amount)` on repaid loans only, so it learns what borrowers like this have shown they can repay rather than copying past lending decisions, including the ones that defaulted.

**Cost matrix.** For a representative GHS 500 loan with an 8% flat fee, 2% funding cost, GHS 5 operating cost, GHS 15 collection cost and 30% recovery, approving a repayer earns G = GHS 25 and approving a defaulter loses L = GHS 380. Declining is worth zero, so the expected value of approving is positive when P(default) < G / (G + L) = 0.062. One default wipes out the profit of about 15 good loans. The threshold moves from 4.5% with no recovery to 16% at 80% recovery, and pricing matters even more: below a 3% fee no loan is profitable. All figures are our assumptions (`src/costs.py`); the BoG Directive requires disclosed, non-compounded interest but sets no cap.

**The same matrix in shillings, from the data** (`13_rubric_additions.ipynb` §13.7). Repaid 7-day Kenyan loans, 96% of the data, carried an average fee of 1.4%: about 70 KES on a 5,000 KES loan, against a loss of about 3,700 KES on a default. At that price the break-even threshold for a 7-day loan is 1.9%, not 6.2%; for 8-to-30-day loans (fee 5.6%) it is 7.3%. Our 6.2% therefore suits a product priced like the Ghana launch loan above and is too lenient for a 7-day loan at Kenyan fees. With thresholds set by term from observed fees, the model approves 87% of applicants instead of 97% and catches 60% of defaults instead of 29%. NkwaPay must supply its real Ghana fee before the threshold is fixed.

# 3. Data and preprocessing (Part B1)

**Loan-level data.** 2,134 loans are co-funded by two lenders and appear on two rows with identical loan details. We merged them into one row per loan with an `n_lenders` feature. At row level the default rate looks like 1.83%; per loan it is 1.63%, because co-funded loans default far more often (8.3% against 1.4%) and were counted twice. Zindi split the competition data by row, so for 920 loans one half is in Train with its label and the other half is in Test. We did not exploit this.

**Distributions.** No values are missing. Amounts range from 2 to 23 million KES, so we use a log scale. 96% of loans are 7-day loans, and longer terms default much more often (52% above 90 days). 22 product types exist; Type_1 is 93% of loans and 11 types have fewer than 30 loans, so rare types are grouped as `Other` and unseen ones as `Unknown` inside each training fold.

**Customer overlap.** 87% of Test customers also appear in Train (all Kenyan); every Ghanaian customer is new. Customers have a median of 6 loans. We therefore never let a customer's loans sit on both sides of a validation split by accident, and we treat a customer's earlier loans as legitimate history.

# 4. Features (Part B2)

All features live in `src/features.py` and are shared by training and the scoring service.

- **Loan terms:** log amount, term, amount per day, lender share, number of co-funders, New/Repeat.
- **Borrower history (14 features):** counts of earlier loans, resolved loans and defaults, smoothed past default rate, active loans and amounts, time since the last loan and the last default, past amounts and terms, distinct lenders.
- **Calendar and macro** features were built and later removed (below).

The timing rules are strict. History counts only loans disbursed on an earlier date, and a past outcome counts only once its due date has passed. Macro indicators for year Y join loans from Y + 1. The brief suggests a backward `merge_asof` without exact matches; we need sums over all earlier loans and a second time key (the due date) for outcomes, so we merge each customer's loans with themselves and keep strictly earlier pairs. A test on made-up loans confirms this equals a backward `merge_asof(allow_exact_matches=False)` on cumulative sums, and fails if same-day matches are let in. A replay test rebuilds the data as it stood on five decision dates and checks that every feature is unchanged; a second test flips all future labels. We injected two leaks by hand to confirm the tests catch them, one of which was a real bug in our first draft (a smoothing prior computed over all labels).

Two findings shaped the model. Only 89 loans have an observable earlier default, because 95% of defaulters never borrow again; the useful history signal is how much and how often someone borrows (6.3% default with no earlier loan, 0.2% after 25 or more). The macro block takes only four distinct values within Kenya, so it can only act as a year label, and Ghana's inflation (up to 31%) lies far outside Kenya's range. It added nothing out of time (at most 0.003 PR-AUC) and was dropped.

# 5. Validation design (Part B3)

We scored logistic regression and LightGBM under three schemes (Table 1). The temporal scheme trains only on loans whose outcome was known before each cut-off and tests on the following window (cut-offs 1 Sep, 1 Oct, 1 Nov and 1 Dec 2022, and 1 Jan 2024).

| Scheme | LogReg PR-AUC | LogReg ROC | LightGBM PR-AUC | LightGBM ROC |
|---|---|---|---|---|
| Stratified 5-fold | 0.184 | 0.860 | **0.233** | 0.889 |
| Grouped by customer | 0.178 | 0.856 | 0.220 | 0.876 |
| Temporal (5 folds) | **0.129** | 0.726 | 0.106 | 0.726 |
| *Naive: row level, leaky columns, random* | | | *0.825* | *0.986* |

: The same models under different validation schemes.

![Scheme comparison](figures/04_scheme_comparison.png){width=90%}

Grouping by customer barely matters (3 to 6% in PR-AUC) because the history features are leakage-safe. Time is what matters: adjusted for each fold's base rate, stratified CV overstates performance 1.7 times for logistic regression and 2.7 times for LightGBM, and it **reverses the ranking of the two models**. In the 2022 folds the models reach ROC about 0.8; from December 2022, when Type_1 disappears, defaults jump and ROC falls to about 0.6. Repeat customers are scored badly in that period too, so the cause is a market change rather than missing history. We used the temporal scheme for everything that follows.

# 6. Models (Parts B4 to B6)

**Classification.** We compared 14 candidates on the temporal folds: a majority-class baseline (it gives everyone the training default rate, so it flags no one and its F1 is 0), four logistic regressions (unweighted, class-weighted, undersampled, stronger regularisation), a decision tree, a random forest (each with and without class weights), three LightGBM variants, k nearest neighbours and an RBF support vector machine. Unweighted logistic regression had the best PR-AUC (0.143) and the best lift after December 2022 (1.81). Shallow LightGBM had the best ROC (0.741). The default LightGBM and the forest fitted the 2022 boom closely and degraded most later. Class weighting did not improve ranking for any model family and destroyed calibration: weighted logistic regression predicts an average default probability of 0.29 against an actual 0.016, which would make a cost threshold meaningless.

**Value in cedis.** Per 1,000 applications, approving everyone earns about GHS 21,000 in the 2022 regime and loses about GHS 16,600 after December 2022. Logistic regression at 0.062 adds little in 2022 (+300) but cuts the later loss to about 3,500. A credit model matters most exactly where it is least accurate.

![Expected value against threshold](figures/05_threshold_value.png){width=95%}

**Calibration.** Probabilities are well calibrated in the 2022 regime (Figure 5). After December 2022 the average is right (10.8% predicted, 10.3% observed) but the spread is too wide: loans scored at 1 to 5% default at 5 to 10%, and loans scored above 50% default at about 22%. Class-weighted logistic regression is far off in every regime.

![Reliability diagrams on the out-of-time folds](figures/05_calibration.png){width=95%}

**Recalibration, out of time** (`13_rubric_additions.ipynb` §13.9, deployed model). We fitted Platt scaling (a slope and intercept on the log-odds) and isotonic regression to the temporal scores. Each fold's calibrator saw only outcomes known before that fold began, from the previous fold alone or from all history; a test checks that no later outcome is used. After December 2022 every variant lowers the Brier score, with bootstrap intervals below zero, and every Platt slope is below 1 (0.30 to 0.74): the spread is fixed. The level is not. Fold 3 (from December 2022) can learn only from boom-period outcomes, so Platt cuts its mean prediction to 5.6% while 12.9% default, approves more loans and loses GHS 9,521 per 1,000 against the raw scores; every variant loses money there. Fold 4, calibrated on 2023 outcomes, gets 9.6% against 8.8% and gains GHS 6,864 per 1,000 with Platt on the latest fold, while all history, swamped by the boom's 1% default rate, barely moves. Pooled over both folds (Table 2), only isotonic on the latest fold has an interval above zero.

| From December 2022 (2,834 loans, 10.3% default) | Brier | Mean P | Approved | GHS per 1,000 against raw (95% interval) |
|--------------------------------|----------|----------|------------|------------------------------------|
| Raw scores | 0.1005 | 10.2% | 69% | (raw: -4,725) |
| Platt, latest fold | 0.0917 | 8.1% | 31% | +949 (-2,497 to 4,297) |
| Isotonic, latest fold | 0.0929 | 7.4% | 61% | +1,891 (71 to 3,907) |
| Platt, all history | 0.0918 | 6.1% | 77% | -1,625 (-3,063 to -313) |
| Isotonic, all history | 0.0930 | 5.7% | 80% | -3,386 (-5,464 to -1,573) |

: Recalibration fitted only on outcomes known before each fold (`reports/13_recalibration.csv`).

![Reliability before and after recalibration](figures/13_recalibration.png){width=85%}

We recommend calibrating only on recent outcomes from the current regime, never the whole history. We tested fold-sized windows, not monthly refits; loans disbursed in a single month of 2023 held as few as 3 defaults, so a monthly refit needs a rolling window long enough to hold about as many defaults as fold 4's (119). We keep Platt as the default, since two parameters are harder to overfit, though this data cannot separate it from isotonic. Because a refit on pre-shift outcomes makes the first window worse, a divergence on the score-band monitor (notebook 12) should trigger human review, not an automatic refit. The shipped scorer still serves raw probabilities.

**Operating point.** Logistic regression, decline when P(default) > 0.062. In 2022 this declines 1% of applicants and catches 15% of defaults; after December 2022 it declines 36% and catches 53%. Pooled F1 is 0.20.

**Confusion matrix and approval goal** (`13_rubric_additions.ipynb`). For the deployed model (calendar features removed, Section 8), at 0.062 the temporal folds give 188 defaulters declined, 467 defaulters approved, 1,135 good borrowers declined and 38,104 good borrowers approved: precision 0.14, recall 0.29, F1 0.19. Stated as a business goal, the cost threshold approves 96.7% of applicants overall, 98.8% in the 2022 regime and 68.6% from December 2022. The approval rate therefore falls automatically when risk rises; if NkwaPay sets a minimum approval rate for inclusion, `reports/13_approval_goal.csv` shows its cost in cedis per 1,000 applications.

![Confusion matrix at the cost threshold](figures/13_confusion_matrix.png){width=100%}

**Threshold sweep** (deployed model, all temporal folds; `reports/13_threshold_sweep.csv`, built by `scripts/threshold_sweep.py` from the saved notebook 13 results).

| Threshold | 0.02 | 0.04 | 0.062 | 0.08 | 0.10 | 0.15 | 0.25 |
|---|---|---|---|---|---|---|---|
| Precision | 0.083 | 0.129 | 0.142 | 0.156 | 0.162 | 0.180 | 0.222 |
| Recall | 0.629 | 0.392 | 0.287 | 0.258 | 0.237 | 0.206 | 0.159 |
| F1 | 0.147 | 0.194 | 0.190 | 0.194 | 0.192 | 0.192 | 0.185 |
| Approval rate | 87.6% | 95.0% | 96.7% | 97.3% | 97.6% | 98.1% | 98.8% |
| GHS per 1,000 | 19,421 | 19,707 | 19,430 | 19,386 | 19,323 | 19,252 | 19,112 |

: Precision, recall and F1 when declining above each threshold.

F1 hardly moves between 0.04 and 0.15 (0.19 throughout) while recall halves, so F1 cannot choose the threshold. The cost matrix does: it sets 0.062 before looking at these folds.

**Regression.** We compared a mean baseline, a persistence rule, linear regression, ridge, lasso, support vector regression and LightGBM. Ridge regression (with terms on a log scale and predictions capped at the largest amount ever repaid) was jointly best with lasso on log RMSE (0.759 and 0.757) and on MAE in KES (about 5,900; median error about 1,930 KES or GHS 150), and slightly better than lasso after December 2022. Support vector regression was clearly worse (0.85). On a raw scale a linear model extrapolated three-year loans to e^57 KES, a useful lesson in why transforms matter. A simple rule, "lend what this borrower borrowed on average before", lands within 25% of the actual amount more often than any model (38%). First-time borrowers are over-predicted by about 40%, so a first-loan cap is needed.

On both scales, out of time, ridge has RMSE 0.759 and MAE 0.550 on the log scale, and RMSE 39,416 KES and MAE 5,907 KES on the original scale; the mean baseline has 1.231, 0.955, 52,135 KES and 9,313 KES (`reports/06_regression_board.csv`). The original-scale RMSE is large because a few loans run to millions of shillings. The residuals (Figure 8) are centred above the bulk of predictions, so large loans are not systematically under-predicted. They have a long negative tail, because predictions rarely fall below about 400 KES while some actual loans are far smaller. Errors are largest for loans above 90 days (log RMSE 1.12) and for borrowers with no history (1.05).

![Ridge residual diagnostics](figures/06_residuals.png){width=95%}

**Does the limit do its job?** (`13_rubric_additions.ipynb` §13.8). With no limit, 18% of the money lent out of time went to borrowers who defaulted, because defaulters ask for much larger loans. Offering the smaller of the request and the ridge limit, with the first-loan cap, cuts the money reaching eventual defaulters by 87% and brings that share to 4.6% (7.2% for the past-average rule). The price is paid by good borrowers: 43% are offered their full request and 64% at least three quarters of it. The limit should be loosened for borrowers with a long clean record.

**Noise floor.** A paired bootstrap that resamples customers (1,000 replicates) gives the following results. Logistic regression beats the baseline on every metric in every replicate. Against the brief's noise floor of ±0.0263, the gains over the baseline (F1 +0.20, PR-AUC +0.10) are far beyond it, while the overall differences from shallow LightGBM (about 0.01 to 0.02) are inside it. Against shallow LightGBM, PR-AUC and F1 are not distinguishable overall; LightGBM is significantly better on ROC (by 0.024) and earns slightly more at the threshold (about GHS 250 per 1,000). After December 2022, logistic regression is significantly better on PR-AUC (+0.038, interval 0.008 to 0.070). The bootstrap also exposed a trap: pooling out-of-fold scores across folds inflates ranking metrics (a constant baseline scores ROC 0.44), so ranking metrics are computed within folds and averaged. We kept logistic regression for its transparency and its edge in the period most like Ghana, and do not claim it is more accurate overall.

| Model | PR-AUC, all | F1, all | PR-AUC, Dec 2022 on | F1, Dec 2022 on |
|---|---|---|---|---|
| Majority class | 0.049 [0.044, 0.056] | 0 | 0.108 [0.093, 0.124] | 0 |
| Logistic regression | 0.143 [0.122, 0.173] | 0.202 [0.178, 0.224] | 0.202 [0.168, 0.250] | 0.236 [0.203, 0.267] |
| Shallow LightGBM | 0.130 [0.112, 0.159] | 0.185 [0.167, 0.203] | 0.165 [0.139, 0.207] | 0.223 [0.195, 0.254] |

: Estimates with 95% bootstrap intervals (1,000 customer-level replicates; F1 at the 0.062 threshold; the Section 5 models, with calendar features; `reports/07_classification_ci.csv`).

# 7. Derivations (LO3)

`notebooks/08_derivations.ipynb` derives the log-loss, gradient $\frac{1}{n}X^\top(p - y)$ and Hessian $\frac{1}{n}X^\top W X$ of logistic regression and shows that the loss is convex. It derives the normal equation $\hat\beta = (X^\top X)^{-1}X^\top y$ and the ridge solution, and gradient boosting with log-loss (pseudo-residuals $y - p$, Newton leaf values $\sum(y-p) / (\sum p(1-p) + \lambda)$). Each is checked on our data: our Newton solver matches scikit-learn to five decimals. Two results connect to the models. The full regression design is rank-deficient (rank 35 of 40 columns), so ridge is needed for a unique solution. And a full Newton boosting step diverges with rare defaults because the curvature $p(1-p)$ is tiny, which is why boosting needs a learning rate.

# 8. Interpretation and reason codes (Part B7)

For a linear model the log-odds split exactly into one contribution per input relative to the average applicant. We sum these into reason families and report the three that push hardest towards decline, in plain language. A test confirms the contributions reproduce the model's probabilities.

Two corrections came out of this. `log_amount` (+1.65) and `amount_per_day_log` (−1.56) nearly cancel and together mostly encode the term, so a "loan too large" reason would have pointed the wrong way; term and amount now form one family. And Ghanaian applicants on unseen products were about to be told their product "had repayment problems", when the model has no record of it at all; they now receive "This product type is new to us, so we have no repayment record for it yet". Calendar features moved individual scores without predicting anything, so the deployed model drops them (PR-AUC 0.1428 to 0.1449, within noise).

# 9. Kenya to Ghana (Part C)

![Kenya and Ghana distributions](figures/10_distributions.png){width=100%}

Ghanaian applications differ on almost every axis: 87% are on products never seen in Kenya, 96% on unseen lenders, 67% are co-funded (3% in Kenya), 14% run longer than 30 days (1% in Kenya), and 21% of borrowers have no earlier loan. A domain classifier separates the countries with AUC 1.000, and still 0.999 without product, lender and co-funding. Part of this is an artefact: Ghanaian borrowers' earlier loans are all unlabelled Test loans, so they look as if they have no repayment record, which would not be true in production.

Importance weighting collapses: the effective sample falls to 1,515 of 66,520 loans (2.3%), and the top 1% of Kenyan loans carry half the weight. The weighted model's Ghana scores correlate only 0.58 with the unweighted ones, and with no Ghanaian outcomes we cannot tell which is better. Applied to Ghana, the model would decline 22% of applicants (Kenya 6.7%): almost every Type_2 and Type_8 loan, but only 8.5% of Type_3, which is 86% of Ghana volume. These are extrapolations, not risk estimates.

Before trusting Ghana scores NkwaPay needs: outcomes from a capped pilot; recalibration and out-of-time validation on those outcomes; definitions of the new products and lenders; full Ghanaian repayment histories; currency-consistent amounts; credit-bureau data; fairness checks on pilot outcomes; and live drift monitoring.

# 10. Ethics and regulation (Part D)

**Segments.** The largest gaps are by borrowing history. Applicants with no earlier loan default at 11% (0.3% for 20+ loans) and are declined 32% of the time, including 30% of those who would have repaid. Decline rates differ by an order of magnitude between lenders and products; the model almost never declines Type_1. Adding PR-AUC, F1 and approval rate by segment (`reports/13_segment_pr_f1.csv`): applicants with no earlier loan are approved 68% of the time against 99.9% for those with 20 or more; the tool is not regressive by loan size, approving 99.6% of the smallest loans and 90.2% of the largest.

**Proxies.** The data holds no protected attributes, so we tested proxies. Borrowing history is the strongest candidate. Removing all six history features changed almost nothing (PR-AUC 0.145 to 0.144; good thin-file borrowers still declined 27% of the time), because missing past amounts and the lack of any repayment record identify the same people. Dropping a proxy does not remove it. Our mitigation is a policy: no automatic decline for applicants without a repayment record, a capped starter loan instead, monthly parity monitoring by history band, and a consented demographic sample kept apart from scoring for audits.

**What the mitigation costs** (`13_rubric_additions.ipynb` §13.5). Without it, 22% of good borrowers with no repayment record would be turned away, against 1.6% of good borrowers who have one. It applies to 665 out-of-time applicants (1.7% of all), 539 of whom repaid. They default 19% of the time, so the first loan loses money at any size: about GHS 12 per affected applicant for a starter loan at 10% of the usual size and GHS 52 for a full-size loan. Across all applications the 10% starter costs about GHS 200 per 1,000, roughly 1% of the portfolio's value. We recommend it as the price of access and of information: 58% of the good borrowers in this group borrowed again.

**Findex 2021 Ghana (survey-weighted).** 60% of adults have a mobile money account, but only 7% borrowed from a formal institution in the past year, while 52% borrowed some money. Account ownership is clearly lower for women (63% against 74%), the poorest 40% (55% against 77%), the least educated and rural adults (57% against 76%). Mobile-money borrowing is not lower for these groups. Digital credit already reaches them, and they arrive with thin formal records, which is exactly the profile the model penalises.

**Regulation.** From the documents we read: the BoG Digital Credit Directive (2025) requires clear disclosure (§6.1, §7.1), complaint resolution within 20 days (§6.1(e)), no harassment in collection (§7.2) and daily, consented reporting to credit bureaus (§7.3). The Data Protection Act 2012 (Act 843) requires minimality (s.19), a lawful basis (s.20), data quality (s.26), access and correction (s.32 to 33) and no processing of special personal data (s.37). Section 41 gives a right to notice and reconsideration within 21 days for decisions taken solely by automated means; s.41(4) exempts decisions about entering a contract, which may cover credit, but we design as if it applies. The Credit Reporting Act 2007 (Act 726) requires a lender to search the applicant's record at a licensed credit bureau before deciding to grant or refuse credit (s.26(4)), to get written consent before sending credit information to a bureau (s.26(1)), and to submit each loan within 72 hours and its repayment status monthly (s.24(3)). It also forbids asking borrowers for race, ethnic origin, political, religious or union affiliation (s.24(4)), which we meet. Our system does none of the bureau steps. The Directive also bars manual approval at the physical office (§2.3(1)), so the human review we offer must run through a digital channel, and it forbids lending one customer more than a limit set by the Bank (§3.5(5)), which our limit model does not yet check. What we cannot meet with the model alone: bureau checks and reporting, Ghana-specific accuracy, group fairness against protected traits, and organisational duties such as registration and breach notification.

The full audit, with every segment table and the complete regulatory mapping, is a separate document (`reports/NkwaPay_Audit_Group3.pdf`). The Datasheet for the data is in `docs/datasheet.md`. The Model Card is in `docs/model_card.md`, and an example decline letter, which offers human review within 21 days, data access and correction, and a 20-day complaint route, is in `docs/decline_letter.md`.

# 11. Scoring service (Part E)

`Scorer.score(application, history)` returns the default probability, a decision, the top three reasons and an offered amount. It uses the same feature code as training; a test shows it rebuilds identical features for random Train loans. The policy has three outcomes: **approve** (offer the smaller of the request and the predicted limit, capped for first loans), **decline** (above the threshold, with a repayment record), and **refer** (thin-file applicants above the threshold, and any market other than Kenya). One rule sits on top of the model: an applicant with an earlier unpaid loan on record is never approved automatically. Only 89 training loans went to such borrowers, 18% of them defaulted (95% interval 11% to 27%), and with so few examples the model approved 29 of 81 of them out of time; a test applicant with three unpaid loans scored 6.1%, just under the threshold (§13.6). A call takes about 20 ms (p95 32 ms), and batch scoring runs at about 115,000 loans per second. The app sends nine application fields and the applicant's earlier loans with outcomes; no personal or sensitive attributes are needed.

# 12. Limitations and next steps

The cost matrix uses our own figures. The default rule and the timing of `Lender_portion_Funded` need confirming with NkwaPay. We read Act 726 as consolidated in 2012; later amendments, the Credit Reporting Regulations 2020 and the Borrowers and Lenders Act 2020 were not reviewed, and none of us is a lawyer. The Kenyan model already struggles after the December 2022 change, and Ghana is further still from its training data. The next step is not a better Kenyan model but Ghanaian outcomes: a small, capped, monitored pilot with human review, followed by recalibration and a Ghana-specific validation.

# 13. Reproducibility

Everything in this report is produced by the code in the repository, with `random_state = 42` throughout and package versions pinned in `requirements.txt`. With the data files in `data/raw/` (sources in `data/README.md`), `make` reruns the thirteen notebooks in order, the tests, the claim check, this report and the slides in about 15 minutes on a laptop. `scripts/verify_claims.py` recomputes 71 of the numbers quoted here from the raw files, independently of the notebooks, and `reports/claims_table.md` lists where each one comes from. The saved model ships in `models/nkwapay-v1.joblib`, and `python scripts/score.py examples/application_repeat.json` scores an application without any training data. The raw data is not in the repository because the Zindi and Findex terms do not allow redistribution.

# 14. AI-use declaration

Declared under the course AI policy (brief, Appendix B3). We used an AI assistant (Claude Code) to scaffold the repository, to draft and debug the notebook, module and script code, to draft report and slide text, and to rerun the notebooks and recheck the quoted numbers.

- Every number in the report comes from a notebook. Seventy-one of them are recomputed from the raw files by `scripts/verify_claims.py` and traced in `reports/claims_table.md`.
- AI-drafted statements that did not match the outputs were corrected, for example the direction of the "loan size" reason code and the share of co-funded Ghana loans.
- The regulatory mapping was written from the legal texts in `docs/legal/`.
- Each member's individual reflection gives their own declaration.
