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

I led three sections. In **problem framing** (`01_problem_framing`, `src/costs.py`) I placed the decision point at approval, defined the label, built the feature audit of every Train column, and built the cost matrix that sets the decline threshold used by the shipped scorer. In **exploratory analysis** (`02_eda`, `src/data.py`) I found and merged the duplicated co-funded loans. In the **Ghana transfer study** (`10_ghana_transfer`) I compared the two markets, trained a domain classifier, tested importance weighting, and listed the evidence NkwaPay needs before trusting Ghana scores.

# What I learned

- **Data leakage: a feature is only valid if it exists when the decision is made.** A model can learn from anything in the table, including values written after the outcome, which quietly hand it the answer. I applied this by drawing the loan timeline and asking, for each column, when its value is set. `Total_Amount_to_Repay` looked like a contract term, but on identical contracts defaulted loans owed far more than repaid ones, because penalties are added after a default. The column was recording the outcome. Kept in, it made the model look excellent (PR-AUC 0.825) while being useless at approval; removed, the honest figure is 0.145.
- **Unit of analysis: every row should be one example of what you predict.** Rates, splits and models all assume this. In the EDA I found that a loan funded by two lenders appears as two rows. Because those loans default more often, counting them twice raised the default rate from 1.63% to 1.83%, and the same loan could fall on both sides of a validation split. Merging them into one row per loan fixed both problems for every later notebook.
- **Cost-sensitive decisions: the threshold comes from money, not from 0.5.** A classifier outputs a probability; the decision rule is a separate choice. Approving a defaulter costs far more than turning away a good borrower, so the lender should decline once the expected loss outweighs the expected gain. I turned this into a cost matrix in cedis, which gave a cut-off of 6.2%. Testing other assumptions showed how fragile one number is: recovery rates moved it between 4.5% and 16%, and at the low fees on Kenyan 7-day loans it fell to 1.9%. Pricing moved the decision more than any choice of model.
- **Distribution shift: a model is only trustworthy where its training data reaches.** Using a Kenyan model in Ghana assumes the two populations look alike. I tested this with a domain classifier, a model trained only to tell Kenyan applications from Ghanaian ones; if it succeeds, they differ. It separated them perfectly, because most Ghana loans use products and lenders never seen in Kenya. Importance weighting tries to fix shift by up-weighting Kenyan loans that resemble Ghana, but it needs overlap, and here about 2% of the data ended up carrying the weight. So I recommended no automated Ghana decisions until a pilot produces local outcomes, and learned that "the model cannot answer this yet" is a valid result.

# What I would do differently

- **Derive the cost matrix from the data first.** I started from assumed Ghana figures (GHS 500 loan, 8% fee); the version from observed Kenyan fees came later and showed one threshold cannot fit every product. I would set thresholds by product and ask NkwaPay for its real Ghana fee in week one.
- **Put open questions to the client early.** The data does not state the default rule, and Ghanaian borrowers' earlier loans are unlabelled, so they look like first-time borrowers. I would put questions like these in the initial action plan instead of finding them midway.
- **Check overlap before modelling the shift.** I would run the unseen-product counts and domain classifier first, and spend the time saved designing the pilot concretely: its size, loan cap, length, and the metrics that would justify going live.
- **Make my work visible in the repository.** Most commits sit under one teammate's account, so the history does not show who did what. Next time I would commit my sections on their own branches and open pull requests for review.

# AI-use declaration

The group's use of an AI assistant (Claude Code) is declared in `docs/ai_use_declaration.md`. I used Claude to help draft this reflection from the repository's README, report and notebooks. I checked each figure against `reports/claims_table.md` and reviewed and edited the text so it reflects my own work and views.
