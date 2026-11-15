---
title: "Individual Reflection: ICS553 Prosit 2 (Group 3)"
geometry: "a4paper, margin=0.65in"
fontsize: 10pt
mainfont: Carlito
monofont: "DejaVu Sans Mono"
pagestyle: empty
---

**Clifford Yeboah** · GitHub: `YCliffe21` · Role: problem framing and cost matrix (Part A), exploratory analysis (Part B1), Kenya-to-Ghana transfer (Part C)

# My contribution

I led three sections of the group's work. In **problem framing** (`01_problem_framing`, `src/schema.py`, `src/costs.py`) I placed the decision point at approval, defined the label, and built the feature audit that classifies every Train column as usable, identifier, redundant or leaky. I built the cost matrix in cedis, which gives the decline threshold P(default) > 0.062 that the shipped scoring service uses. In **exploratory analysis** (`02_eda`, `src/data.py`) I found that co-funded loans appear on two rows and merged them to one row per loan, which every later notebook relies on. For the **Ghana transfer study** (`10_ghana_transfer`) I compared the Kenyan and Ghanaian distributions, trained a domain classifier, tested importance weighting, and wrote the list of evidence NkwaPay needs before it can trust Ghana scores.

# What I learned

- **Leakage hides in columns that look legitimate.** `Total_Amount_to_Repay` reads like a contract term. But on identical contracts, repaid loans owe about 0.7% over principal and defaulted loans 41%, because penalties are added after a default. With that column in, a naive pipeline scores PR-AUC 0.825; our honest out-of-time estimate is 0.145. I now ask when a column was last written, not only what it means.
- **The unit of analysis changes the headline.** The brief's 1.83% default rate is per row. Per loan it is 1.63%, because co-funded loans default far more often (8.3% against 1.4%) and had been counted twice. Checking what one row represents has to come before any modelling.
- **The threshold is a business number before it is a model number.** One default wipes out the profit of about 15 good loans. The threshold moves from 4.5% with no recovery to 16% at 80% recovery. When the matrix was later rebuilt from the fees actually charged in the Kenyan data (1.4% on a 7-day loan), the break-even threshold for 7-day loans was 1.9%, not 6.2%. The pricing assumptions moved the decision more than any choice of model.
- **Some distribution shift cannot be corrected.** 87% of Ghana loans are on products never seen in Kenya and 96% on unseen lenders, and the domain classifier separated the two countries with AUC 1.000. Importance weighting left an effective sample of 1,515 of 66,520 loans (2.3%), and its Ghana scores correlated only 0.58 with the unweighted model's. Reweighting needs overlap, and there was almost none. The honest recommendation was no automated Ghana decisions until a capped pilot produces outcomes. I learned that "the model cannot answer this yet" is a legitimate result to report.

# What I would do differently

- **Derive the cost matrix from the data first.** I started from assumed Ghana figures (GHS 500 loan, 8% fee). The shilling version from observed fees came later and showed that one threshold cannot fit every product. I would set thresholds by product from the start and ask NkwaPay for its real Ghana fee in week one.
- **Put the open questions to the client early.** The data does not state the default rule (days past due or grace period). Ghanaian borrowers' earlier loans are all unlabelled Test loans, so they look like borrowers with no repayment record. I would list questions like these in the initial action plan instead of finding them midway through the work.
- **Check overlap before modelling the shift.** I would run the unseen-product counts and the domain classifier first, then decide whether importance weighting is worth trying. I would spend the time saved designing the pilot concretely: its size, loan cap and length, and the metrics that would justify going live.
- **Make my work visible in the repository.** Most of the commit history sits under one teammate's account, so it does not show who did what. Next time I would commit my sections on their own branches (`section/01-framing`, `section/02-eda`, `section/10-transfer`) and open pull requests for review.

# AI-use declaration

The group's use of an AI assistant (Claude Code) is declared in `docs/ai_use_declaration.md`. I used Claude to help draft this reflection from the repository's README, report and notebooks. I checked each figure against `reports/claims_table.md` and reviewed and edited the text so it reflects my own work and views.
