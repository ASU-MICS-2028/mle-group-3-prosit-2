---
title: "Group Reflection: ICS553 Prosit 2 (Group 3)"
geometry: "a4paper, margin=0.65in"
fontsize: 10pt
mainfont: Carlito
pagestyle: empty
---

**Group 3** · Mubarak Tijani, Eric Elikplim Sunu, Clifford Yeboah, Adzeyena Opei · November 2026

The PBL return stage asks the group one question: did we find a solution? This page answers it, says who led what, and records what we would change as a group. Each member's individual reflection covers their own learning.

# Did we find a solution?

**For Kenya, yes.** NkwaPay asked for a tool that estimates default risk, suggests a loan size and explains each decision. We deliver an unweighted logistic regression that declines when P(default) > 0.062, a threshold set from a cedi cost matrix; a ridge loan limit trained on repaid loans; three plain-language reasons per decision; and a scoring service that runs from the saved model with no training data (`models/nkwapay-v1.joblib`). Its honest out-of-time PR-AUC is 0.145. A naive pipeline that used the leaky `Total_Amount_to_Repay` column scored 0.825 (report, Section 1).

**For Ghana, not yet, and we think that is the right answer.** 87% of Ghana loans are on products never seen in Kenya and 96% on unseen lenders, and a classifier separates the two countries with AUC 1.0 (report, Section 9). We recommend a capped pilot with human review and no automated Ghana decisions until Ghanaian outcomes exist.

# Who led what

| Member | Led |
|---|---|
| Mubarak Tijani | Repository and pipeline; features, validation, classification and the operating point; scoring service; number check |
| Eric Elikplim Sunu | Loan-limit regression; noise floor; derivations; reason codes |
| Clifford Yeboah | Framing and cost matrix; exploratory analysis; Kenya to Ghana transfer |
| Adzeyena Opei | Ethics and regulation; Model Card, Datasheet and decline letter; report and slides |

# What worked

- **Checks anyone can rerun.** 39 tests, including a leakage replay, and `scripts/verify_claims.py`, which recomputes 71 of the numbers we quote from the raw files (`reports/claims_table.md`).
- **Validating on time.** Adjusted for base rate, stratified cross-validation overstated performance 1.7 times for logistic regression and 2.7 times for LightGBM, and reversed their ranking; testing on later loans picked logistic regression (report, Section 5).
- **Putting money on errors.** The cost matrix turned model quality into a decision in cedis, and later showed how strongly the threshold depends on the fee NkwaPay charges.

# What we would do differently

- **Spread the work and make it visible.** Of the 27 commits on `main` before 19 November (merges excluded), 26 are under one account, and every section was first committed on 29 October from that account. The table above credits four leads; the history cannot show it. Next time each lead works on a section branch and opens a pull request that another member reviews.
- **Test a recommendation before we write it.** "Recalibrate monthly" sat in our report and Model Card from 29 October. When we tested recalibration out of time on 19 November, fitting on outcomes from before the December 2022 shift made the first window worse (GHS -9,521 per 1,000) and only the next window gained (GHS +6,864; claims C70, C71). We changed the advice to recent outcomes only, with human review after a shift.
- **Put the client questions in the action plan.** The default rule and NkwaPay's real Ghana fee stayed open to the end. At the fee Kenya charged on 7-day loans (1.4%), the break-even threshold for those loans would be 1.9%, not 6.2%. We would ask both in the first week.

# AI use

We used Claude Code throughout, to scaffold, draft, debug and recheck. The declaration is in the report (Section 14) and `docs/ai_use_declaration.md`. Statements that did not match the outputs were corrected, and every member must be able to explain the work without AI.
