# Datasheet: data used for the NkwaPay credit scoring model (ICS553 Prosit 2)

Structured after Gebru et al. (2021), *Datasheets for Datasets*. It describes the data as we received and used it. Where the original publisher does not say something, we write "not stated" rather than guess.

## Motivation
- **Purpose.** The African Credit Scoring Challenge on Zindi (competition from Nov 2024 to Jan 2025) released the loan data to build default-prediction models that generalise across countries. We use it to score NkwaPay applications and to judge transfer from Kenya to Ghana.
- **Who created it and who funded it.** Published by Zindi. The original lender or lenders, and how the loans were collected, are not stated.

## Composition
| File | What one row is | Size we checked |
|---|---|---|
| Train.csv | one lender's share of one loan, with outcome | 68,654 rows, 16 columns, 66,520 distinct loans, 6,540 customers, all Kenya |
| Test.csv | the same, without outcome | 18,594 rows (15,069 Kenya, 3,525 Ghana); Ghana has 2,084 loans from 446 customers |
| economic_indicators.csv | one country and one indicator, yearly columns | 27 rows, 9 indicators, Ghana, Kenya and Côte d'Ivoire |
| micro_gha.csv (Global Findex 2021, Ghana) | one survey respondent | 1,000 respondents, 120 columns, survey weight `wgt` |

- **Label.** `target` = 1 if the loan defaulted, 0 if repaid. 1,258 defaulting rows (1.83%); 1.63% per loan after merging co-funded rows. The rule for "default" (for example days past due) is not stated.
- **Time span.** Train loans were disbursed from 4 Oct 2021 to 14 Nov 2024. Ghana Test loans run from 30 Nov 2021 to 21 Mar 2023.
- **Missing values.** None in Train or Test.
- **Duplicates and structure.** A loan funded by two lenders appears on two rows (2,134 such loans in Train). We merge them to one row per loan (`src/data.py`). 920 loans have one row in Train and the other in Test, and 4,306 Test customers also appear in Train. We did not exploit either.
- **Categories.** 22 loan types and 4 lenders in Train. Type_1 is 93% of loans. Ghana uses loan types and lenders almost entirely absent from Train.
- **Sensitive data.** No names, phone numbers, gender, age, income, location or other special personal data (Act 843 s.37). Customers appear only as numeric IDs.
- **Errors or noise found.** `Total_Amount_to_Repay` (and `Lender_portion_to_be_repaid`, derived from it) is updated after default, so it records the outcome. We drop both. Amounts are in local currency, so Kenyan (KES) and Ghanaian (GHS) values are not directly comparable.

## Collection process
- Not stated by the publisher: how applicants were selected, which loans were approved, and whether declined applications exist. **The data only contains approved loans**, so any model trained on it learns from borrowers a past policy already accepted (selection bias).
- Macro indicators are described by Zindi as taken from FRED; the indicator codes Zindi lists (for example FP.CPI.TOTL.ZG) are World Bank series codes.
- Findex 2021 is a nationally representative World Bank survey; estimates must use the `wgt` weights.

## Preprocessing we applied
- Merge co-funded rows into loans; drop leaky, redundant and ID columns (feature audit, Section 1).
- Log transform of amounts; borrower history built only from earlier loans and only from outcomes already due (Section 3).
- Rare loan types grouped as Other, unseen as Unknown, inside each training fold.
- The raw files are kept unchanged in `data/raw/`; every step is reproducible with `scripts/run_all.sh`.

## Uses
- **Used for.** Training and validating the default classifier and the loan-size regression; describing Kenya to Ghana shift; Findex gap analysis.
- **Should not be used for.** Automated decisions about Ghanaian applicants without Ghanaian outcomes; any decision about individuals outside credit assessment; inferring protected traits.
- **Known risks.** Selection bias (approved loans only), a market change in Dec 2022 (Type_1 ends), and thin-file borrowers being under-represented (567 "New Loan" rows in Train).

## Distribution and maintenance
- Zindi data under the competition rules; Findex under the World Bank Microdata Library terms, which forbid redistribution. Neither is committed to this repository (`data/README.md`).
- Maintained by ICS553 Group 3 for this course only. No updates planned.
