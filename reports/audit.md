---
title: "NkwaPay Credit Scoring: Ethical and Regulatory Audit"
subtitle: "ICS553 Prosit 2, Group 3"
author:
  - Mubarak Tijani
  - Eric Elikplim Sunu
  - Clifford Yeboah
  - Adzeyena Opei
date: "November 2026"
---

This audit covers Part D of the brief: the Model Card, the fairness results by segment, the proxy analysis and mitigation, the mapping to Ghanaian regulation, the Findex access analysis and the decline letter. The text and tables are generated from `notebooks/11_ethics_regulation.ipynb`, `notebooks/13_rubric_additions.ipynb` and `reports/*.csv` by `scripts/build_audit.py`. The Datasheet for the data is in `docs/datasheet.md`.

# 1. Model Card


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
- **Drift:** performance roughly halved after Dec 2022. Recalibrate only on recent outcomes from the current regime, in a window holding about 100 defaults, never the whole history (Section 6; monthly refits were not tested). After a shift, send a divergence on the score-band monitor to human review rather than refitting automatically. Monitor approval rate, default rate and mean prediction.
- **Regulation:** no credit-bureau integration, so the bureau search required before a credit decision (Act 726 s.26(4)) and bureau reporting (Act 726 s.24; BoG Directive §7.3) are not met. Act 1052 not reviewed from primary text.
- Cost-matrix figures are assumptions; replace with NkwaPay's own.


# 2. Results by segment

All figures are for the deployed logistic regression on the out-of-time folds, at the 6.2% threshold. The ratio column is PR-AUC divided by the segment's default rate.

| segment | group | loans | defaults | default rate | PR-AUC | ratio | F1 | approval rate |
|---|---|---|---|---|---|---|---|---|
| country | Kenya | 39,894 | 655 | 0.016 | 0.149 | 9.074 | 0.190 | 0.967 |
| regime | 2022 boom | 37,060 | 364 | 0.010 | 0.104 | 10.538 | 0.126 | 0.988 |
| regime | Dec 2022 onward | 2,834 | 291 | 0.103 | 0.197 | 1.920 | 0.234 | 0.686 |
| loan type | Other | 794 | 117 | 0.147 | 0.426 | 2.888 | 0.351 | 0.402 |
| loan type | Type_1 | 36,363 | 299 | 0.008 | 0.029 | 3.567 | 0.000 | 0.999 |
| loan type | Type_4 | 613 | 67 | 0.109 | 0.188 | 1.724 | 0.203 | 0.176 |
| loan type | Type_5 | 478 | 33 | 0.069 | 0.067 | 0.977 | 0.105 | 0.592 |
| loan type | Type_7 | 1,646 | 139 | 0.084 | 0.130 | 1.535 | 0.113 | 0.934 |
| lender | 251804 | 2,452 | 197 | 0.080 | 0.096 | 1.192 | 0.139 | 0.710 |
| lender | 267277 | 260 | 72 | 0.277 | 0.582 | 2.101 | 0.443 | 0.027 |
| lender | 267278 | 37,182 | 386 | 0.010 | 0.080 | 7.723 | 0.142 | 0.990 |
| New vs Repeat | New Loan | 205 | 22 | 0.107 | 0.350 | 3.266 | 0.273 | 0.571 |
| New vs Repeat | Repeat Loan | 39,689 | 633 | 0.016 | 0.143 | 8.984 | 0.185 | 0.969 |
| loan size band | Q1 smallest | 7,981 | 121 | 0.015 | 0.051 | 3.355 | 0.013 | 0.996 |
| loan size band | Q2 | 7,978 | 55 | 0.007 | 0.051 | 7.325 | 0.093 | 0.991 |
| loan size band | Q3 | 7,977 | 131 | 0.016 | 0.099 | 6.042 | 0.121 | 0.981 |
| loan size band | Q4 | 7,979 | 100 | 0.013 | 0.067 | 5.331 | 0.135 | 0.964 |
| loan size band | Q5 largest | 7,979 | 248 | 0.031 | 0.268 | 8.620 | 0.268 | 0.902 |
| borrowing history | 0 earlier loans | 1,781 | 195 | 0.109 | 0.286 | 2.610 | 0.270 | 0.678 |
| borrowing history | 1-4 | 4,808 | 189 | 0.039 | 0.137 | 3.487 | 0.168 | 0.891 |
| borrowing history | 5-19 | 18,121 | 226 | 0.012 | 0.055 | 4.437 | 0.106 | 0.988 |
| borrowing history | 20+ | 15,184 | 45 | 0.003 | 0.013 | 4.429 | 0.033 | 0.999 |
| all | all | 39,894 | 655 | 0.016 | 0.149 | 9.074 | 0.190 | 0.967 |

Section 11.2 reported decline rate, precision and recall by segment. The brief also asks for PR-AUC and F1. Selection rate here is the approval rate.

PR-AUC within a segment is computed on scores pooled across folds. Section 7 showed pooling across folds inflates ranking metrics, so compare segments with each other, not with the headline 0.145. A segment needs at least 10 defaults for PR-AUC.

**Reading.** The ratio column (PR-AUC divided by the segment's default rate) is the fairer comparison, because PR-AUC rises with the base rate. Country has a single row because every labelled loan is Kenyan; Ghana's predicted decline rate (22%) is in Section 10 and is not a performance metric.

Decline rate, recall, false positive rate and precision for the same segments:

| segment | group | decline_rate | recall (defaults caught) | good borrowers declined (FPR) | precision |
|---|---|---|---|---|---|
| country | Kenya | 0.033 | 0.287 | 0.029 | 0.142 |
| regime | 2022 boom | 0.012 | 0.137 | 0.010 | 0.116 |
| regime | Dec 2022 onward | 0.314 | 0.474 | 0.296 | 0.155 |
| loan type | Other | 0.598 | 0.889 | 0.548 | 0.219 |
| loan type | Type_1 | 0.001 | 0.000 | 0.001 | 0.000 |
| loan type | Type_4 | 0.824 | 0.866 | 0.819 | 0.115 |
| loan type | Type_5 | 0.408 | 0.364 | 0.411 | 0.062 |
| loan type | Type_7 | 0.066 | 0.101 | 0.063 | 0.128 |
| lender | 251804 | 0.290 | 0.320 | 0.287 | 0.089 |
| lender | 267277 | 0.973 | 1.000 | 0.963 | 0.285 |
| lender | 267278 | 0.010 | 0.137 | 0.008 | 0.147 |
| New vs Repeat | New Loan | 0.429 | 0.682 | 0.399 | 0.170 |
| New vs Repeat | Repeat Loan | 0.031 | 0.273 | 0.027 | 0.140 |
| loan size band | Q1 smallest | 0.004 | 0.008 | 0.004 | 0.029 |
| loan size band | Q2 | 0.009 | 0.109 | 0.009 | 0.081 |
| loan size band | Q3 | 0.019 | 0.130 | 0.017 | 0.113 |
| loan size band | Q4 | 0.036 | 0.260 | 0.033 | 0.092 |
| loan size band | Q5 largest | 0.098 | 0.556 | 0.083 | 0.177 |
| borrowing history | 0 earlier loans | 0.322 | 0.533 | 0.296 | 0.181 |
| borrowing history | 1-4 | 0.109 | 0.317 | 0.101 | 0.114 |
| borrowing history | 5-19 | 0.012 | 0.102 | 0.010 | 0.110 |
| borrowing history | 20+ | 0.001 | 0.022 | 0.001 | 0.067 |
| all | all | 0.033 | 0.287 | 0.029 | 0.142 |

**Reading.**

- **Country.** Every labelled loan is Kenyan, so no Ghana accuracy can be reported. Section 10 gives Ghana's *predicted* decline rate (22%), which is not a performance metric.
- **New vs Repeat and borrowing history** show the largest gaps. Borrowers with no earlier loans default far more often, and the model declines them far more often, including many who would have repaid (a high false-positive rate). On the positive side, recall is also highest there: the model is doing what the data says, but the cost of being a thin-file borrower falls on good thin-file borrowers too.
- **Lender and loan type.** Decline rates differ by an order of magnitude between lenders and products. The model almost never declines Type_1 (0.1%), which is 91% of these loans; nearly all declines fall on the other products, where defaults are 7 to 15%. Part of this is genuine product risk, part is lender book composition, which a borrower cannot control.
- **Loan size.** The largest-loan quintile is declined most, consistent with term and size driving risk.
- Segments flagged *small sample* (fewer than 30 defaults) have unstable rates; treat them as indicative.
- The ROC-AUC column is computed on out-of-fold scores pooled across folds, which Section 7 showed inflates ranking metrics (pooled 0.835 against the fold average of 0.725). Use it only to compare segments with each other.

# 3. Proxy analysis and mitigation

**Direct protected attributes.** The data contains none of the "special personal data" of Act 843 s.37(1) (religion, ethnic origin, race, trade-union membership, political opinions, health, sexual life, criminal behaviour), nor gender, age, income or location. So the model cannot use them directly, **and we cannot measure fairness against them directly** either.

**Candidate proxies.** A feature can stand in for a protected trait if it is strongly associated with it in the population:

| Feature | Could proxy for | Basis |
|---|---|---|
| Borrowing history (`n_prev_loans`, `is_new`, `has_resolved_history`) | Gender, income, education, age, rural residence | Findex 2021 Ghana (11.5): formal and digital borrowing histories are unevenly distributed across these groups |
| Loan amount and past loan sizes | Income | Larger loans go to higher earners |
| Lender and product | Region, channel, occupation | Lenders and products serve different customer bases |

**Test.** The borrowing-history family is the strongest proxy candidate and drives the largest gap in 11.2. We measure what happens if we remove it from the model.

**Reading.** Removing the six history features **changes almost nothing**: PR-AUC moves from 0.145 to 0.144, and good borrowers with no earlier loans are still declined about 27% of the time (against 30%), while good borrowers with 20+ loans are almost never declined. The history signal flows back through correlated features: missing past-loan sizes, no recent borrowing, and the absence of any resolved record all identify a thin-file applicant just as well. **Dropping a proxy does not remove it.** Thin-file borrowers genuinely default more in this data (11% against 0.3%), so any accurate model will treat them differently; the question is what NkwaPay does with that score.

**Mitigation we recommend: a thin-file policy rather than a blind model.**

1. Keep history in the model (it reflects real repayment behaviour), but **do not auto-decline** applicants with no repayment record. Route them to a **starter-loan track**: a small capped first loan (the Section 6 regression over-predicts first loans by about 40%, so a cap is needed anyway), with limits that step up after on-time repayment. This builds the history the model lacks, which matters most in Ghana (11.5).
2. **Monitor parity**: track decline rate and good-borrower decline rate (FPR) by history band and New vs Repeat each month, with a review trigger if the FPR ratio exceeds an agreed level.
3. **Collect consented, lawful attributes for auditing only** (gender, age band, region) on a sample, kept separate from the scoring pipeline, so true group fairness can be measured, consistent with Act 843 s.19 (minimality) and s.20 (consent).

## What the mitigation costs

Section 11.3 proposed a mitigation: an applicant with no repayment record who scores above the threshold is not declined automatically but offered a small starter loan. Here we measure who that affects and what it costs under the cost matrix. A starter loan is priced like any other loan (`src/costs.py`), with the principal scaled down.

**Reading.** Without the mitigation, 22% of good borrowers with no repayment record would be turned away, against 1.6% of good borrowers who have one. The mitigation applies to 665 applicants, 539 of whom repaid.

It is not free. These applicants default 19% of the time, far above the 6.2% break-even, so every loan size loses money on the first loan: about GHS 12 per affected applicant for a 10% starter loan and GHS 52 for a full-size one. Spread over all applications, the 10% starter costs about GHS 200 per 1,000 applications, roughly 1% of the portfolio's value, and a full-size loan costs four times as much. The small starter loses money even when repaid, because the GHS 5 operating cost is larger than its fee.

So the starter loan is a price NkwaPay pays for access and for information: 58% of the good borrowers in this group came back for another loan, and each repaid starter gives the next decision a repayment record. Our recommendation stays a starter loan of about 10%, not a full-size loan and not an automatic decline. The cost figure depends on our fee and cost assumptions.

| policy for these applicants | GHS per affected applicant | GHS per 1,000 applications | share of portfolio value |
|---|---|---|---|
| decline all of them | 0.000 | 0.000 | 0.000 |
| starter loan at 10% of the usual size (GHS 50) | -12.232 | -203.890 | -0.010 |
| starter loan at 20% of the usual size (GHS 100) | -16.621 | -277.059 | -0.014 |
| starter loan at 50% of the usual size (GHS 250) | -29.789 | -496.566 | -0.026 |
| full-size loan | -51.737 | -862.410 | -0.044 |

# 4. Regulatory mapping

Section numbers are from the documents in `docs/legal/`. Act 726 places its duties on a "financial institution licensed and regulated by law" (s.61); we read that as covering a BoG-licensed digital credit provider such as NkwaPay.

| Requirement | What it asks | How our design responds | Status |
|---|---|---|---|
| BoG Directive §6.1(a), §7.1(3) | Clear, comprehensive, accurate information; disclose charges, interest, total cost of credit, due date | Decline letter (11.6) and reason codes state the decision and its main factors in plain language; the cost model uses disclosed flat, non-compounded pricing | Partly. Pricing disclosure is a product task outside the model |
| BoG Directive §6.1(b) | Digital pre-agreement: principal, disbursement schedule, cost of default, fees, insurance | Scoring service returns the approved limit; pre-agreement content is outside the model | Not met by the model; product must supply it |
| BoG Directive §6.1(c), §7.1(3)(b) | Interest disclosed for the tenure, not compounded | Cost matrix (src/costs.py) uses a flat fee rate; no compounding | Met in our assumptions |
| BoG Directive §6.1(e) | Complaint receipt and resolution within twenty days | Decline letter gives a complaint channel and the 20-day commitment | Met in the letter; NkwaPay must run the process |
| BoG Directive §7.2 | No harassment, threats, or contacting a customer's contacts during collection | The model uses no contact-list or social data, and recovery assumptions do not rely on such tactics | Met for the model |
| BoG Directive §7.3(5)-(9) | Submit customer credit information to licensed bureaus daily, with prior customer consent | Not in scope of the model; bureau data is recommended as a future feature (Part C) | Not met: no bureau integration in our system |
| BoG Directive §2.3(1) | The physical office must not be used for manual onboarding, approval, disbursement, collection or recovery | Scoring is fully digital. The human review we offer on declines and referrals must also run through a digital channel, not at the office | Met for scoring; the review step needs a digital workflow |
| BoG Directive §3.5(5), §7.1(2) | No digital credit to a single customer above the limit set by the Bank; credit limits in the credit policy must comply with BoG requirements | The limit model caps offers at the largest amount ever repaid and caps first loans (Section 13.8), but checks no BoG limit; the figure is not in the Directive | Not met: the BoG limit must be added as a hard cap |
| BoG Directive §2.1 (licence application, items x and xi) | A Data Protection Impact Assessment approved by the Data Protection Commission, and a data protection certificate | The Model Card, Datasheet and this audit are inputs to an impact assessment; none has been submitted | Not met: organisational |
| Act 843 s.17-18 | Accountability, lawful and reasonable processing, purpose specification, openness, security, participation | Model Card documents purpose, data and limits; features are restricted to loan application and repayment records | Partly. Security and accountability processes are organisational |
| Act 843 s.19 | Minimality: processing necessary, relevant and not excessive | Calendar and macro features removed (no predictive value); leaky columns dropped; no special or social data used | Met for model inputs |
| Act 843 s.20, s.22-23 | Consent or lawful basis; specific purpose; data subject made aware | Credit assessment is necessary for the contract (s.20(1)(a)); notice of purpose is in the letter | Partly. Consent capture is a product task |
| Act 843 s.26 | Quality of information: complete, accurate, up to date | Leakage audit (Section 1) found Total_Amount_to_Repay updated after default; excluded | Met for inputs we audited |
| Act 843 s.32-33, s.35 | Access to personal data; correction of inaccurate data | Letter tells the applicant how to request their data and correct errors; reason codes show which records drove the decision | Met in the letter; NkwaPay must handle requests |
| Act 843 s.37 | Special personal data (religion, ethnicity, race, health, politics, etc.) not processed | None used; proxies checked in 11.3 | Met; proxy risk monitored |
| Act 843 s.41 | Automated decisions that significantly affect a person: notify, and reconsider on request within 21 days; s.41(4)(a)-(b) exempts decisions about entering into a contract | Credit approval arguably falls under the contract exemption, but we design as if s.41 applies: the letter notifies the automated basis and offers human reconsideration within 21 days | Met by design (stronger than the minimum) |
| Act 726 s.26(4) | Search the applicant's credit record at one or more licensed credit bureaus before deciding to grant or refuse credit | The scorer uses NkwaPay's own loan records only and does not query a bureau | Not met: a bureau search must happen before the score is used for a decision |
| Act 726 s.26(1)-(2), (6)-(8), s.35 | Written borrower consent before credit information goes to a bureau, requested on application forms and loan agreements and not induced; consent records kept for at least ten years | Outside the model; the scoring service needs no consent field | Not met by the model; consent capture and storage are product tasks |
| Act 726 s.24(1)-(3) | Submit loan details to a licensed bureau within 72 hours of the credit agreement, and repayment status monthly | No bureau reporting in our system (BoG Directive §7.3 asks for daily submission) | Not met: no bureau integration in our system |
| Act 726 s.24(4) | Do not require a borrower's race, ethnic origin, political, religious or union affiliation | None of these is collected or used (Section 11.3 checks proxies) | Met |
| Act 726 s.25 | Loans ninety days past due may be reported without consent only if the debt is undisputed, no satisfactory repayment proposal was made and 28 days' notice was given | The data does not state the default rule (Section 1), so we cannot say which of our defaults would be reportable | Not met: needs NkwaPay's days-past-due definition |
| Act 726 s.23, s.32(1) | Credit reports are confidential, used only to evaluate credit risk, obtained with the subject's written consent and shown to the subject on request | No bureau data used yet | Not applicable until bureau integration |
| Act 726 s.37, s.41 | Right to see and challenge credit information held about you; court action for harm from inaccurate information | Decline letter offers access to and correction of the data NkwaPay holds; bureau-held records are outside our system | Partly: met for our own data only |
| Act 843 s.36 / Act 726 s.30 | Credit-bureau data subject rights; a bureau keeps credit information for at most six years (ten for a criminal conviction) | No bureau data used yet | Not applicable until bureau integration |
| BoG Directive §7.1(1) | Digital credit policy in line with the Borrowers and Lenders Act 2020 (Act 1052) | Not reviewed | Unknown; Act 1052 not reviewed |

**What we cannot meet with the model alone.**

1. **Credit bureau checks and reporting.** BoG Directive §7.3 requires daily, consented submission to licensed bureaus, and Act 726 requires a bureau search before any decision to grant or refuse credit (s.26(4)), written consent (s.26(1)), submission within 72 hours and monthly status updates (s.24(3)). Our system has no bureau link. Act 726 s.21(p) also tells bureaus not to mark a person down merely because they hold no credit information about them, which is the same principle as our thin-file mitigation in 11.3.
2. **Ghana-specific validity.** No Ghanaian outcomes exist, so we cannot show the model is accurate or fair for Ghanaian applicants (Part C), which undermines the "accurate information" duties for Ghana.
3. **Group fairness against protected traits.** Without lawful, consented demographic data we can only test proxies.
4. **Organisational duties** (registration as a data controller, s.27; security, s.28; breach notification, s.31; complaint handling, Directive §6.1(e)) depend on NkwaPay's processes, not the model.
5. **Act 1052** (Borrowers and Lenders Act 2020), amendments to Act 726 after 2012 and the Credit Reporting Regulations 2020 (L.I. 2394) were not reviewed from their texts. We are not lawyers; the whole mapping should be confirmed by one.

# 5. Debt and access: Findex 2021 Ghana

| group | level | n | Any account | Financial institution account | Mobile money account | Borrowed any money (12 months) | Borrowed from a formal institution | Borrowed via mobile money |
|---|---|---|---|---|---|---|---|---|
| all | all | 1000 | 0.68 | 0.39 | 0.60 | 0.52 | 0.07 | 0.07 |
| gender | men | 535 | 0.74 | 0.47 | 0.65 | 0.57 | 0.09 | 0.05 |
| gender | women | 465 | 0.63 | 0.32 | 0.55 | 0.48 | 0.06 | 0.08 |
| income | poorest 40% | 309 | 0.55 | 0.25 | 0.46 | 0.52 | 0.05 | 0.07 |
| income | richest 60% | 691 | 0.77 | 0.49 | 0.69 | 0.52 | 0.09 | 0.06 |
| education | primary or less | 295 | 0.52 | 0.24 | 0.43 | 0.50 | 0.05 | 0.09 |
| education | secondary | 638 | 0.80 | 0.47 | 0.71 | 0.53 | 0.07 | 0.05 |
| education | tertiary | 64 | 0.98 | 0.95 | 0.91 | 0.62 | 0.32 | 0.06 |
| age_band | 15-24 | 291 | 0.65 | 0.28 | 0.62 | 0.50 | 0.04 | 0.03 |
| age_band | 25-34 | 375 | 0.76 | 0.48 | 0.70 | 0.53 | 0.07 | 0.08 |
| age_band | 35-49 | 226 | 0.67 | 0.38 | 0.59 | 0.58 | 0.09 | 0.08 |
| age_band | 50+ | 104 | 0.65 | 0.46 | 0.47 | 0.47 | 0.10 | 0.07 |
| area | rural | 360 | 0.57 | 0.27 | 0.48 | 0.52 | 0.07 | 0.07 |
| area | urban | 640 | 0.76 | 0.48 | 0.68 | 0.52 | 0.08 | 0.07 |

**Reading (survey-weighted, 95% intervals from the Kish effective sample size; about 1,000 respondents, so subgroup gaps of a few points are within noise).**

- **Mobile money is widespread, formal borrowing is rare.** 60% of adults have a mobile money account, but only 7% borrowed from a formal institution and 7% via mobile money in the past year, while 52% borrowed *some* money, mostly informally. Demand for credit is broad; formal records are thin.
- **Account gaps are large and clear:** women 63% against men 74%; poorest 40% at 55% against 77%; primary education or less 52% against 80% for secondary; rural 57% against 76%.
- **Formal borrowing** is lower for women (5.6% against 9.0%), the poorest (4.9% against 8.8%), the least educated and the youngest, though most of these intervals overlap.
- **Borrowing via mobile money is not lower for these groups**; the point estimates are slightly *higher* for women, the poorest and the least educated (within noise). Digital credit is already reaching them, and they arrive with little formal history.
- **Link to the model.** The deployed model's strongest proxy is borrowing history (11.3), and it declines a third of applicants with no earlier loans. In Ghana, the people most likely to be new to formal credit are women, poorer, less educated, younger and rural adults, and they are already digital-credit users. Without the thin-file policy, the model would likely decline these groups disproportionately even though no protected attribute is in it.
- **Caveat.** Findex describes the adult population, not NkwaPay's applicants, and it measures access, not repayment. It shows where the proxy risk lies; only pilot data with consented demographics can show the actual effect.

# 6. Decline letter


Dear Customer,

Thank you for applying for a loan with NkwaPay. We are sorry to tell you that we cannot approve this application at this time.

**Why.** Your application was assessed by an automated scoring system that estimates how likely a loan is to be repaid on time, using the details of this application and your past loans with us. It does not use your religion, ethnicity, health, political opinions or any other sensitive personal information. The main reasons for this decision were:

- Loans of this product type have had higher repayment problems.
- Your recent borrowing pattern (time since your last loan and loans currently running) differs from that of borrowers who usually repay.
- The sizes of your previous loans are in a range where repayment problems have been more common.

**What you can do.**
- **Ask for a person to review this decision.** Reply to this message or call our support line within 21 days and a member of our team, not the automated system, will look at your application again and tell you in writing what we will do.
- **Try a smaller or shorter loan.** Shorter repayment periods and smaller first loans are approved more often, and repaying on time helps future applications.
- **Check the information we hold about you.** You can ask to see the personal data we used and ask us to correct anything that is wrong or out of date.
- **Make a complaint.** If you are unhappy with how we handled your application, contact our complaints team. We will resolve your complaint within 20 days. You can also contact the Data Protection Commission or the Bank of Ghana.

This decision is about this application only. It is not a judgement about you, and it does not stop you applying again.

Yours sincerely,
NkwaPay Customer Care

*Reference: application 104034 (model score 8%; internal use)*


**Design choices.** Short sentences and everyday words; the decision first; reasons in the applicant's terms (from `src/reasons.py`, never calendar or sensitive data); concrete next steps. The 21-day human review reflects Act 843 s.41(2)-(3) (applied even though the contract exemption in s.41(4) may cover credit decisions), access and correction reflect s.32-33, and the 20-day complaint commitment reflects BoG Directive §6.1(e). For Ghana applicants with an unseen product, the reason reads "This product type is new to us, so we have no repayment record for it yet", which is true and invites review rather than implying a bad record. The letter should be offered in English and the main local languages, and by SMS in a shortened form with a link to the full text.
