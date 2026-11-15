#!/usr/bin/env python
"""Assemble the ethical and regulatory audit (Part D) into one document.

    python scripts/build_audit.py        -> reports/audit.md

Nothing is written by hand here: the text comes from notebook 11 and docs/, and
the tables from reports/*.csv, so the audit cannot drift from the analysis.
scripts/build_report.sh turns reports/audit.md into Word and PDF.
"""
import json
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REPORTS, DOCS = ROOT / "reports", ROOT / "docs"


def notebook_markdown(name: str) -> list:
    cells = json.loads((ROOT / "notebooks" / name).read_text())["cells"]
    return ["".join(c["source"]) for c in cells if c["cell_type"] == "markdown"]


def section(cells: list, heading: str) -> str:
    """Markdown cells from the one starting with `heading` up to the next '## ' heading, heading line dropped."""
    start = next(i for i, c in enumerate(cells) if c.lstrip().startswith(heading))
    out = [cells[start].split("\n", 1)[1] if "\n" in cells[start] else ""]
    for c in cells[start + 1:]:
        if c.lstrip().startswith("## "):
            break
        out.append(c)
    text = "\n\n".join(s.strip() for s in out if s.strip())
    return re.sub(r"(?m)^(?![-|\s]|\d+\. )(.+)\n(?=(- |\d+\. ))", r"\1\n\n", text)   # a list needs a blank line before it


def demote(md: str, levels: int = 1) -> str:
    return re.sub(r"^(#+) ", lambda m: "#" * (len(m.group(1)) + levels) + " ", md, flags=re.M)


def table(df: pd.DataFrame, digits: int = 3) -> str:
    df = df.copy()
    for c in df.columns:
        if pd.api.types.is_float_dtype(df[c]):
            whole = (df[c].dropna() % 1 == 0).all()
            df[c] = df[c].map(lambda v: "" if pd.isna(v) else (f"{v:,.0f}" if whole else f"{v:.{digits}f}"))
    head = "| " + " | ".join(map(str, df.columns)) + " |\n|" + "---|" * len(df.columns)
    rows = ["| " + " | ".join("" if pd.isna(v) else str(v).replace("|", "/") for v in r) + " |" for r in df.itertuples(index=False)]
    return "\n".join([head] + rows)


def main() -> None:
    nb11, nb13 = notebook_markdown("11_ethics_regulation.ipynb"), notebook_markdown("13_rubric_additions.ipynb")
    seg = pd.read_csv(REPORTS / "13_segment_pr_f1.csv")
    seg = seg[["segment", "group", "loans", "defaults", "default rate", "PR-AUC", "PR-AUC / default rate", "F1", "selection (approval) rate"]]
    seg = seg.rename(columns={"PR-AUC / default rate": "ratio", "selection (approval) rate": "approval rate"})
    seg11 = pd.read_csv(REPORTS / "11_segment_metrics.csv")
    seg11 = seg11[["segment", "group", "decline_rate", "recall (defaults caught)", "good borrowers declined (FPR)", "precision"]]
    findex = pd.read_csv(REPORTS / "11_findex_gaps.csv")
    findex = findex[[c for c in findex.columns if not c.endswith("±")]]
    card = (DOCS / "model_card.md").read_text().split("\n", 1)[1]
    letter = (DOCS / "decline_letter.md").read_text().split("\n", 1)[1]

    parts = [
        '---\ntitle: "NkwaPay Credit Scoring: Ethical and Regulatory Audit"\nsubtitle: "ICS553 Prosit 2, Group 3"\n'
        "author:\n  - Mubarak Tijani\n  - Eric Elikplim Sunu\n  - Clifford Yeboah\n  - Adzeyena Opei\n"
        'date: "November 2026"\n---',
        "This audit covers Part D of the brief: the Model Card, the fairness results by segment, the proxy analysis and "
        "mitigation, the mapping to Ghanaian regulation, the Findex access analysis and the decline letter. The text and "
        "tables are generated from `notebooks/11_ethics_regulation.ipynb`, `notebooks/13_rubric_additions.ipynb` and "
        "`reports/*.csv` by `scripts/build_audit.py`. The Datasheet for the data is in `docs/datasheet.md`.",
        "# 1. Model Card", card,
        "# 2. Results by segment",
        "All figures are for the deployed logistic regression on the out-of-time folds, at the 6.2% threshold. "
        "The ratio column is PR-AUC divided by the segment's default rate.",
        table(seg), section(nb13, "## 13.3"),
        "Decline rate, recall, false positive rate and precision for the same segments:", table(seg11),
        section(nb11, "## 11.2"),
        "# 3. Proxy analysis and mitigation", section(nb11, "## 11.3"),
        "## What the mitigation costs", section(nb13, "## 13.5"),
        table(pd.read_csv(REPORTS / "13_mitigation_cost.csv"), 3),
        "# 4. Regulatory mapping", section(nb11, "## 11.4").split("**What we cannot meet")[0].strip(),
        table(pd.read_csv(REPORTS / "11_regulatory_mapping.csv")),
        "**What we cannot meet" + section(nb11, "## 11.4").split("**What we cannot meet", 1)[1],
        "# 5. Debt and access: Findex 2021 Ghana", table(findex, 2), section(nb11, "## 11.5"),
        "# 6. Decline letter", letter, section(nb11, "## 11.6"),
    ]
    (REPORTS / "audit.md").write_text("\n\n".join(p for p in parts if p.strip()) + "\n")
    print("Wrote reports/audit.md")


if __name__ == "__main__":
    main()
