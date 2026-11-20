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

from src.policy import Policy, record_review  # noqa: E402
from src.scoring import Scorer  # noqa: E402

EXAMPLES = {
    "Repeat borrower, good record": "application_repeat.json",
    "Large 90-day loan, earlier default": "application_decline.json",
    "Earlier unpaid loans": "application_past_default.json",
    "First loan (no record)": "application_first_loan.json",
    "Ghana applicant": "application_ghana.json",
}
# Product types: the data gives codes only, no descriptions. These figures are from the
# Kenyan training loans (one row per loan) and the Test applications; see notebooks 02 and 10.
# (loans in Kenya, default rate, median amount, median term in days, Ghana applications)
PRODUCTS = {
    "Type_1": (61723, 0.0084, 4930, 7, 0), "Type_7": (1646, 0.0844, 5906, 7, 0),
    "Type_5": (1166, 0.0832, 18000, 14, 0), "Type_4": (872, 0.1284, 30000, 30, 0),
    "Type_10": (319, 0.0282, 101347, 29, 0), "Type_6": (255, 0.1176, 4490, 7, 0),
    "Type_9": (205, 0.1756, 20200, 30, 0), "Type_14": (99, 0.7172, 10000, 30, 0),
    "Type_2": (74, 0.6081, 150000, 136, 274), "Type_11": (42, 0.0476, 8114, 30, 0),
    "Type_18": (30, 0.0333, 2950, 30, 0), "Type_12": (21, 0.0476, 15600, 14, 0),
    "Type_20": (13, 0.5385, 202730, 30, 0), "Type_17": (13, 0.0, 12286, 7, 0),
    "Type_23": (12, 0.75, 1169944, 90, 0), "Type_13": (10, 0.0, 500, 30, 0),
    "Type_16": (6, 0.0, 24472, 11, 0), "Type_15": (4, 0.75, 15000000, 183, 0),
    "Type_24": (3, 0.0, 15000, 7, 0), "Type_21": (3, 0.0, 289000, 30, 0),
    "Type_19": (3, 0.0, 515665, 912, 0), "Type_22": (1, 0.0, 46080, 15, 0),
    "Type_3": (0, None, None, None, 1782), "Type_8": (0, None, None, None, 28),
}
N_KENYA = 66520
NEW_PRODUCT = "New product (not listed)"


def product_label(code):
    if code == NEW_PRODUCT:
        return code
    n, rate, *_ = PRODUCTS[code]
    if n == 0:
        return f"{code} · Ghana only, unknown to the model"
    tag = "" if n >= 30 else " · rare"
    return f"{code} · {n:,} Kenyan loans · {rate:.1%} defaulted{tag}"


def product_note(code):
    if code == NEW_PRODUCT:
        return "A product the model has never seen. It is scored as unknown, with no product information."
    n, rate, amount, term, ghana = PRODUCTS[code]
    if n == 0:
        return (f"Never offered in Kenya; it appears only in {ghana:,} Ghana applications. "
                "The model has no repayment record for it and scores it as unknown.")
    text = (f"{n:,} Kenyan loans ({n / N_KENYA:.1%} of all), {rate:.1%} of them defaulted. "
            f"Typical loan {amount:,.0f} KES over {term:.0f} days.")
    if n < 30:
        text += " Too few loans to learn on its own, so the model groups it with the other rare products."
    else:
        text += " The model learned this product separately."
    if ghana:
        text += f" Also used in {ghana:,} Ghana applications."
    return text

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
        "loan_type": app["loan_type"] if app["loan_type"] in PRODUCTS else NEW_PRODUCT,
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
    st.session_state.pop("scored", None)


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
    st.divider()
    st.subheader("Lending policy")
    st.caption("Amounts NkwaPay sets. Changing them needs no retraining and never changes the risk or the decision.")
    kenya_cap = st.number_input("First-loan cap, Kenya (KES)", min_value=1.0, value=float(scorer.starter_cap),
                                step=500.0, help="Largest first loan. Default: median repaid first loan in the data.")
    starter_pct = st.slider("Starter loan for referrals (% of usual offer)", 1, 50, 10,
                            help="Report section 13.5 costs a 10% starter loan at about 1% of portfolio value.")
    ghana_cap = st.number_input("First-loan cap, Ghana (GHS, 0 = not set)", min_value=0.0, value=0.0, step=50.0,
                                help="Ghana has no default: a reviewer can only offer a pilot loan once this is set.")
    kenya_max = st.number_input("Maximum per customer, Kenya (KES, 0 = none)", min_value=0.0, value=0.0,
                                step=1000.0, help="For a limit set by the regulator (BoG Directive s.3.5(5)).")
policy = Policy(first_loan_cap={"Kenya": kenya_cap, **({"Ghana": ghana_cap} if ghana_cap > 0 else {})},
                starter_share=starter_pct / 100, max_loan={"Kenya": kenya_max} if kenya_max > 0 else {})

left, right = st.columns([3, 2], gap="large")

with left:
    st.subheader("Application")
    c1, c2, c3 = st.columns(3)
    c1.selectbox("Country", ["Kenya", "Ghana"], key="country_id")
    c2.selectbox("Product type", list(PRODUCTS) + [NEW_PRODUCT], key="loan_type", format_func=product_label,
                 help="The data gives product codes only, not descriptions. Each option shows how common the "
                      "product was in Kenya and how often its loans defaulted.")
    c3.radio("Borrower", ["New Loan", "Repeat Loan"], key="repeat", horizontal=True)
    st.caption(f"**{st.session_state['loan_type']}:** {product_note(st.session_state['loan_type'])}")
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

if go:
    application = {
        "customer_id": st.session_state["customer_id"],
        "country_id": st.session_state["country_id"],
        "loan_type": "Unknown" if st.session_state["loan_type"] == NEW_PRODUCT else st.session_state["loan_type"],
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
        past = [{k: (None if pd.isna(v) else v) for k, v in r.items()} for r in rows.to_dict("records")]
        for r in past:
            pd.Timestamp(r["disbursement_date"]), pd.Timestamp(r["due_date"])
        st.session_state["scored"] = {"application": application, "result": scorer.score(application, past, policy)}
    except Exception as exc:  # show input problems instead of a stack trace
        st.session_state.pop("scored", None)
        st.session_state["score_error"] = str(exc)
    else:
        st.session_state.pop("score_error", None)

with right:
    st.subheader("Decision")
    if st.session_state.get("score_error"):
        st.error(f"Could not score this application: {st.session_state['score_error']}. "
                 "Check the dates and numbers in the earlier loans.")
    elif "scored" not in st.session_state:
        st.info("Fill in the application (or load an example) and press **Score application**.")
    else:
        application = st.session_state["scored"]["application"]
        result = st.session_state["scored"]["result"]
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

        review = result.get("review")
        if review:
            st.divider()
            st.markdown("**Reviewer step**")
            ceiling = review["max_amount"]
            if ceiling is None:
                st.warning(review["note"] + " Use the Ghana cap in the sidebar, then score again.")
            else:
                st.caption(f"Suggested starter loan: **{ceiling:,.0f} {review['currency']}**. The reviewer may "
                           "lower it or decline (0), never raise it, and must give a reason. Each decision is logged.")
                with st.form("review_form", clear_on_submit=False):
                    amount = st.number_input(f"Amount to offer ({review['currency']})", min_value=0.0,
                                             max_value=float(ceiling), value=float(ceiling), step=50.0)
                    reviewer = st.text_input("Reviewer", placeholder="e.g. reviewer-07")
                    reason = st.text_area("Reason", placeholder="e.g. mobile-money income confirmed; full starter loan")
                    submitted = st.form_submit_button("Record decision", use_container_width=True)
                if submitted:
                    try:
                        rec = record_review(result, application, amount, reviewer, reason)
                    except ValueError as exc:
                        st.error(str(exc))
                    else:
                        st.session_state.setdefault("review_log", []).append(rec)
                        what = (f"Starter loan of {rec['offered_amount']:,.0f} {rec['currency']}"
                                if rec["outcome"] == "starter_loan" else "Declined by the reviewer")
                        st.success(f"Recorded: {what}.")

        st.caption(f"Scored in {result['latency_ms']:.0f} ms · model {result['model_version']}")
        with st.expander("Full response (JSON)"):
            st.json(result)

if st.session_state.get("review_log"):
    st.subheader("Review log (this session)")
    log = pd.DataFrame(st.session_state["review_log"])
    st.dataframe(log[["time", "customer_id", "market", "p_default", "suggested_amount", "offered_amount", "currency",
                      "outcome", "lowered", "reviewer", "reason"]], use_container_width=True, hide_index=True)
    st.download_button("Download log (JSON lines)", "\n".join(log.apply(lambda r: r.to_json(), axis=1)),
                       file_name="review_log.jsonl", mime="application/json")
