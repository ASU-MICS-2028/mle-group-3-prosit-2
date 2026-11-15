# Claims table

Every number below is quoted in the report, the README or the slides. The "Source" column says which notebook produces it, and the ID is the check in `scripts/verify_claims.py` that recomputes it from the raw files. Run `make verify` to recheck all of them; the last run reproduced 56 of 56.

Numbers that are not in this table (for example the bootstrap intervals, the regression errors and the domain-classifier AUC) come straight from the notebook named beside them in the report and are saved in `reports/*.csv`. They are not independently recomputed.

| ID | Claim | Quoted value | Source |
|---|---|---|---|
| C01 | Rows in Train.csv | 68,654 | `01_problem_framing` |
| C02 | Train rows that defaulted | 1,258 | `01_problem_framing` |
| C03 | Default rate per row | 1.83% | `02_eda` §2.1 |
| C04 | Train loans after merging co-funded rows | 66,520 | `02_eda`, `src/data.py` |
| C05 | Default rate per loan | 1.63% | `02_eda` |
| C06 | Co-funded loans in Train | 2,134 | `02_eda` |
| C07 | Default rate of co-funded loans | 8.3% | `02_eda` |
| C08 | Default rate of single-lender loans | 1.4% | `02_eda` |
| C09 | Loans with one row in Train and the other in Test | 920 | `02_eda` |
| C10 | Test customers who also appear in Train | 4,306 | `02_eda` |
| C11 | Share of Test customers also in Train | 87% | `02_eda` |
| C12 | Ghana Test customers who appear in Train | 0 | `02_eda` |
| C13 | AUC of `Total_Amount_to_Repay / Total_Amount` against default | 0.98 | `01_problem_framing` §1.3 |
| C14 | Amount owed over principal, repaid loans on the same contract | 0.7% | `01_problem_framing` §1.3 |
| C15 | Amount owed over principal, defaulted loans on the same contract | 41% | `01_problem_framing` §1.3 |
| C16 | Rows where `due_date` is not `disbursement_date + duration` | 0 | `01_problem_framing` |
| C17 | Share of loans with a 7-day term | 96% | `02_eda` |
| C18 | Loan types in Train | 22 | `02_eda` |
| C19 | Type_1 share of loans | 93% | `02_eda` |
| C20 | Loan types with fewer than 30 loans | 11 | `02_eda` |
| C21 | Ghana Test loans | 2,084 | `10_ghana_transfer` |
| C22 | Ghana loans on a product never seen in Kenya | 87% | `10_ghana_transfer` §10.1 |
| C23 | Ghana loans with a lender never seen in Kenya | 96% | `10_ghana_transfer` §10.1 |
| C24 | Ghana loans that are co-funded | 67% | `10_ghana_transfer` §10.1 |
| C25 | Gain from a repaid GHS 500 loan | GHS 25 | `01_problem_framing`, `src/costs.py` |
| C26 | Loss from a defaulted GHS 500 loan | GHS 380 | `01_problem_framing`, `src/costs.py` |
| C27 | Decline threshold on P(default) | 0.062 | `01_problem_framing`, `src/costs.py` |
| C28 | Threshold with no recovery | 4.5% | `01_problem_framing` |
| C29 | Threshold with 80% recovery | 16% | `01_problem_framing` |
| C30 | PR-AUC of the deployed model, temporal folds | 0.145 | `09_reason_codes`, `13_rubric_additions` |
| C31 | ROC-AUC of the deployed model, temporal folds | 0.725 | `09_reason_codes`, `13_rubric_additions` |
| C32 | Loans scored out of time | 39,894 | `13_rubric_additions` |
| C33 | Defaulters declined | 188 | `13_rubric_additions` |
| C34 | Defaulters approved | 467 | `13_rubric_additions` |
| C35 | Good borrowers declined | 1,135 | `13_rubric_additions` |
| C36 | Good borrowers approved | 38,104 | `13_rubric_additions` |
| C37 | Precision at the threshold | 0.14 | `13_rubric_additions` |
| C38 | Recall at the threshold | 0.29 | `13_rubric_additions` |
| C39 | F1 at the threshold | 0.19 | `13_rubric_additions` |
| C40 | Approval rate, all folds | 96.7% | `13_rubric_additions` |
| C41 | Approval rate, 2022 regime | 98.8% | `13_rubric_additions` |
| C42 | Approval rate, from December 2022 | 68.6% | `13_rubric_additions` |
| C43 | Decline threshold stored in the shipped model | 0.062 | `12_scoring_service` |
| C44 | Loans the shipped model was trained on | 66,520 | `12_scoring_service` |
| C45 | Shipped model against a fresh refit, largest gap in P(default) | 0 | `12_scoring_service` |
| C46 | Ghana adults with a mobile money account (Findex 2021, weighted) | 60% | `11_ethics_regulation` §11.5 |
| C47 | Ghana adults who borrowed from a formal institution | 7% | `11_ethics_regulation` §11.5 |
| C48 | Ghana adults who borrowed any money | 52% | `11_ethics_regulation` §11.5 |
| C49 | Account ownership, women | 63% | `11_ethics_regulation` §11.5 |
| C50 | Account ownership, men | 74% | `11_ethics_regulation` §11.5 |
| C51 | Good borrowers with no repayment record who score above the threshold | 22% | `13_rubric_additions` §13.5 |
| C52 | Thin-file applicants above the threshold, out of time | 665 | `13_rubric_additions` §13.5 |
| C53 | Of those, repaid | 539 | `13_rubric_additions` §13.5 |
| C54 | Train loans made after an earlier default was on record | 89 | `13_rubric_additions` §13.6 |
| C55 | Default rate of those loans | 18% | `13_rubric_additions` §13.6 |
| C56 | Such applicants approved by the model alone, out of time | 29 | `13_rubric_additions` §13.6 |
