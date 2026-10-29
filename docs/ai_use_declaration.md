# AI-use declaration (draft for the team to review and sign)

**Course:** ICS553, Prosit 2 (NkwaPay credit scoring) · **Group:** 3 · **Date:** November 2026

## Tool used
Claude Code (Anthropic), an AI coding assistant running in the terminal with access to the project repository.

## What it was used for
- **Code.** Drafting the shared modules in `src/` (data loading, features, models, validation, bootstrap, reason codes, scoring service), the tests in `tests/`, and the notebook-building scripts; running them and fixing errors.
- **Analysis.** Running the experiments in notebooks 01 to 12 and drafting the interpretation text in each notebook.
- **Writing.** Drafting the report (`reports/report.md`), the slide deck, the Model Card, the decline letter and this declaration.
- **Research support.** Searching for and downloading the Data Protection Act 2012 (Act 843) text, and reading the BoG Digital Credit Directive. It could not obtain the Credit Reporting Act 2007 (Act 726); obligations under that Act are marked unverified throughout.

## How outputs were checked
- Every number in the notebooks, report and slides is produced by code in this repository and can be regenerated with `scripts/run_all.sh`.
- 26 automated tests check data shapes, the feature audit, leakage (replay and label-flip tests, plus two deliberately injected leaks that the tests caught), the loan-level merge, the bootstrap, reason codes (exact reconstruction of model probabilities) and the scoring service (identical features to training).
- Several AI-drafted claims were found wrong when checked against outputs and were corrected, among them: the direction of the "loan size" reason code, the calibration diagnosis after December 2022, a residual-spread claim, and pooled versus per-fold ranking metrics in the bootstrap. Interpretation text was revised to match the numbers.
- **Each group member must still review the work they present**, rerun the notebooks, and be able to explain every number (workplan item 13).

## What was not delegated
- The problem framing choices (decision point, regression target on repaid loans, cost-matrix assumptions) and the final recommendations are the group's responsibility and should be confirmed or changed by the group before submission.
- Individual reflections (item 14) must be written by each member.

## Statement
We used an AI assistant as described above. We have reviewed its outputs, checked them against the code and data, and take responsibility for the submitted work.

Signed: ______________________ (each member)
