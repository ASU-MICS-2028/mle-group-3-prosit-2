"""NkwaPay loan scorer: a small web front end for the shipped model.

    streamlit run app/app.py

Uses the same Scorer as scripts/score.py (src/scoring.py, models/nkwapay-v1.joblib),
so the answers match the command-line scorer. No training data is needed.
"""
import json
import sys
import warnings
from datetime import date
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore", category=FutureWarning)

from src.scoring import Scorer  # noqa: E402

EXAMPLES = {
    "Repeat borrower, good record": "application_repeat.json",
    "Large 90-day loan, earlier default": "application_decline.json",
    "Earlier unpaid loans": "application_past_default.json",
    "First loan (no record)": "application_first_loan.json",
    "Ghana applicant": "application_ghana.json",
}
HISTORY_COLS = ["disbursement_date", "due_date", "Total_Amount", "duration", "lender_id", "target"]
DECISION_STYLE = {
    "approve": ("Approve", "#2E7D4F"),
    "decline": ("Decline", "#AA3C3F"),
    "refer": ("Refer to a person", "#B7791F"),
}

st.set_page_config(page_title="NkwaPay Loan Scorer", page_icon="💳", layout="wide")


@st.cache_resource
def load_scorer():
    return Scorer.load()


def load_example(name):
    req = json.loads((ROOT / "examples" / EXAMPLES[name]).read_text())
    app = req["application"]
    hist = pd.DataFrame(req.get("history") or [], columns=HISTORY_COLS)
    st.session_state.update({
        "customer_id": int(app["customer_id"]),
        "country_id": app["country_id"],
        "loan_type": app["loan_type"],
        "lender_id": int(app["lender_id"]),
        "n_lenders": max(1, len(app.get("lenders") or [1])),
        "share": float(sum(l.get("share") or 0 for l in (app.get("lenders") or [])) or 1.0),
        "amount": float(app["Total_Amount"]),
        "duration": int(app["duration"]),
        "application_date": date.fromisoformat(app["application_date"]),
        "repeat": app["New_versus_Repeat"],
        "history": hist,
    })
    st.session_state.pop("history_editor", None)


scorer = load_scorer()
if "amount" not in st.session_state:
    load_example("Repeat borrower, good record")

st.markdown(
    "<h1 style='color:#AA3C3F;margin-bottom:0'>NkwaPay Loan Scorer</h1>",
    unsafe_allow_html=True,
)
st.caption(
    "ICS553 Prosit 2 · Group 3 · Logistic regression trained on 66,520 Kenyan loans. "
    f"Declines when P(default) > {scorer.threshold:.1%} (cost-matrix threshold). "
    "A class project on public competition data, not a real lending service."
)

with st.sidebar:
    st.subheader("Load an example")
    choice = st.selectbox("Example application", list(EXAMPLES), label_visibility="collapsed")
    if st.button("Load example", use_container_width=True):
        load_example(choice)
        st.rerun()
    st.divider()
    st.markdown(
        "**How it decides**\n\n"
        "- **Approve** below the threshold; the offer is the smaller of the request and the predicted limit.\n"
        "- **Decline** above the threshold for applicants with a repayment record, "
        "or anyone with an earlier unpaid loan.\n"
        "- **Refer** first-time applicants above the threshold (starter loan or human review), "
        "and every non-Kenyan application, because the model is not validated outside Kenya."
    )

left, right = st.columns([3, 2], gap="large")

with left:
    st.subheader("Application")
    c1, c2, c3 = st.columns(3)
    c1.selectbox("Country", ["Kenya", "Ghana"], key="country_id")
    c2.text_input("Product type", key="loan_type", help="e.g. Type_1. Unseen types are treated as unknown.")
    c3.radio("Borrower", ["New Loan", "Repeat Loan"], key="repeat", horizontal=True)
    c1, c2, c3 = st.columns(3)
    c1.number_input("Amount requested", min_value=1.0, step=500.0, key="amount",
                    help="Local currency (KES for Kenya, GHS for Ghana).")
    c2.number_input("Term (days)", min_value=1, max_value=720, step=1, key="duration")
    c3.date_input("Application date", key="application_date")
    c1, c2, c3 = st.columns(3)
    c1.number_input("Lender ID", min_value=0, step=1, key="lender_id")
    c2.number_input("Number of lenders", min_value=1, max_value=5, step=1, key="n_lenders")
    c3.number_input("Share funded by lenders", min_value=0.0, max_value=1.0, step=0.05, key="share",
                    help="Total share of the loan funded by the lender(s), 0 to 1.")

    st.subheader("Earlier loans")
    st.caption("One row per earlier loan. Outcome: 0 = repaid, 1 = defaulted, blank = not yet due.")
    history = st.data_editor(
        st.session_state["history"],
        num_rows="dynamic",
        use_container_width=True,
        key="history_editor",
        column_config={
            "disbursement_date": st.column_config.TextColumn("Disbursed (YYYY-MM-DD)"),
            "due_date": st.column_config.TextColumn("Due (YYYY-MM-DD)"),
            "Total_Amount": st.column_config.NumberColumn("Amount", min_value=0),
            "duration": st.column_config.NumberColumn("Term (days)", min_value=1, step=1),
            "lender_id": st.column_config.NumberColumn("Lender ID", step=1),
            "target": st.column_config.NumberColumn("Outcome", min_value=0, max_value=1, step=1),
        },
    )
    go = st.button("Score application", type="primary", use_container_width=True)

with right:
    st.subheader("Decision")
    if not go:
        st.info("Fill in the application (or load an example) and press **Score application**.")
    else:
        application = {
            "customer_id": st.session_state["customer_id"],
            "country_id": st.session_state["country_id"],
            "loan_type": st.session_state["loan_type"].strip() or "Unknown",
            "lender_id": int(st.session_state["lender_id"]),
            "lenders": [{"lender_id": int(st.session_state["lender_id"]),
                         "share": float(st.session_state["share"]) / int(st.session_state["n_lenders"])}]
            * int(st.session_state["n_lenders"]),
            "Total_Amount": float(st.session_state["amount"]),
            "duration": int(st.session_state["duration"]),
            "application_date": st.session_state["application_date"].isoformat(),
            "New_versus_Repeat": st.session_state["repeat"],
        }
        rows = history.dropna(how="all")
        try:
            past = [
                {k: (None if pd.isna(v) else v) for k, v in r.items()}
                for r in rows.to_dict("records")
            ]
            for r in past:
                pd.Timestamp(r["disbursement_date"]), pd.Timestamp(r["due_date"])
            result = scorer.score(application, past)
        except Exception as exc:  # show input problems instead of a stack trace
            st.error(f"Could not score this application: {exc}. Check the dates and numbers in the earlier loans.")
            st.stop()

        label, colour = DECISION_STYLE.get(result["decision"], (result["decision"], "#383838"))
        st.markdown(
            f"<div style='background:{colour};color:white;padding:14px 18px;border-radius:8px;"
            f"font-size:1.6rem;font-weight:700'>{label}</div>",
            unsafe_allow_html=True,
        )
        if result.get("note"):
            st.caption(result["note"])

        m1, m2 = st.columns(2)
        m1.metric("Chance of default", f"{result['p_default']:.1%}",
                  f"threshold {result['threshold']:.1%}", delta_color="off")
        offer = result.get("offered_amount")
        m2.metric("Amount offered", "—" if offer is None else f"{offer:,.0f}",
                  f"requested {result['requested_amount']:,.0f}", delta_color="off")
        st.progress(min(result["p_default"] / max(2 * result["threshold"], 1e-9), 1.0),
                    text="Risk relative to the threshold (middle of the bar = threshold)")

        if result["reasons"]:
            st.markdown("**Main reasons**")
            for i, r in enumerate(result["reasons"], 1):
                st.markdown(f"{i}. {r['text']}")
        else:
            st.caption("No reasons are given for an approval.")
        st.caption(f"Scored in {result['latency_ms']:.0f} ms · model {result['model_version']}")
        with st.expander("Full response (JSON)"):
            st.json(result)
