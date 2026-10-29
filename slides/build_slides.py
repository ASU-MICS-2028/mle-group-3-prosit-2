"""Build the debrief deck: python slides/build_slides.py -> slides/NkwaPay_Debrief_Group3.pptx"""
import math
from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.util import Inches, Pt

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "reports" / "figures"
INK, MUTED, ACCENT = RGBColor(0x1F, 0x29, 0x37), RGBColor(0x6B, 0x72, 0x80), RGBColor(0xC2, 0x41, 0x0C)

prs = Presentation()
prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)
BLANK = prs.slide_layouts[6]
FITS = []


def text(slide, s, x, y, w, h, size=18, bold=False, color=INK):
    tf = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h)).text_frame
    tf.word_wrap = True
    for i, line in enumerate(s.split("\n")):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = line
        p.font.size, p.font.bold, p.font.color.rgb = Pt(size), bold, color
        p.space_after = Pt(6)
    return tf


def fit_size(lines, width_in, max_h_in, size=18, floor=12):
    """Largest font size whose estimated wrapped height fits the box."""
    while size > floor:
        cpl = width_in * 72 / (size * 0.52)
        n = sum(math.ceil(len(l) / cpl) for l in lines)
        if n * size * 1.22 / 72 + len(lines) * 6 / 72 <= max_h_in:
            break
        size -= 1
    return size


def slide(title, bullets=None, image=None, takeaway=None, img_w=7.4):
    s = prs.slides.add_slide(BLANK)
    text(s, title, 0.6, 0.4, 12.1, 0.9, size=30, bold=True)
    if bullets:
        width = 12.1 if image is None else 12.1 - img_w - 0.3
        lines = [f"• {b}" for b in bullets]
        box_h = 4.8 if takeaway else 5.6
        size = fit_size(lines, width, box_h)
        FITS.append((title, size))
        text(s, "\n".join(lines), 0.6, 1.5, width, box_h, size=size)
    if image:
        x = 0.6 if not bullets else 13.333 - 0.6 - img_w
        s.shapes.add_picture(str(FIG / image), Inches(x), Inches(1.6), width=Inches(img_w))
    if takeaway:
        text(s, takeaway, 0.6, 6.55, 12.1, 0.7, size=17, bold=True, color=ACCENT)
    return s


s = prs.slides.add_slide(BLANK)
text(s, "NkwaPay credit scoring", 0.8, 2.3, 11.5, 1.2, size=44, bold=True)
text(s, "A Kenyan model, a Ghanaian launch, and what stands between them", 0.8, 3.4, 11.5, 0.8, size=24, color=MUTED)
text(s, "ICS553 Prosit 2 · Group 3 · 3 November 2026", 0.8, 4.5, 11.5, 0.6, size=18, color=MUTED)

slide("The question",
      ["NkwaPay wants to score Ghanaian loan applications",
       "We have 66,520 labelled Kenyan loans (1.63% default) and 2,084 unlabelled Ghanaian ones",
       "Two tasks: will this loan default? how much should we lend?",
       "Decision point: approval, before disbursement; nothing known after it may be a feature",
       "Cost matrix (GHS 500 loan): a good loan earns 25, a default loses 380 → decline if P(default) > 0.062"],
      image="01_loan_timeline.png", img_w=6.4,
      takeaway="One default wipes out the profit of 15 good loans.")

slide("The data gives the answer away, if you let it",
      ["Total_Amount_to_Repay looks like a contract term",
       "On identical contracts, repaid loans owe +0.7%; defaulted loans owe +41% (penalties added later)",
       "Alone it predicts default with AUC 0.98",
       "A naive pipeline scores PR-AUC 0.825; our honest estimate is 0.145",
       "Also: co-funded loans appear twice; Zindi split by row, so 920 loans straddle Train and Test"],
      image="01_repay_ratio_leakage.png", img_w=6.6,
      takeaway="We dropped it. A model using it would fail on every real applicant.")

slide("Validation design picks the model",
      ["Stratified CV: LightGBM 0.233 vs LogReg 0.184 PR-AUC",
       "Temporal CV: LogReg 0.129 vs LightGBM 0.106",
       "Grouping by customer barely matters (history features are leakage-safe)",
       "Time is what matters: random CV overstates 1.7–2.7×"],
      image="04_scheme_comparison.png", img_w=7.2,
      takeaway="Random CV doesn't just inflate the score, it picks the wrong model.")

slide("Kenya changed in December 2022",
      ["Type_1, 93% of all loans and ~1% default, stops after 1 Dec 2022",
       "Afterwards: 9–13% default, mostly new customers",
       "Both models fall from ROC ~0.8 to ~0.6",
       "Repeat customers are scored badly too: the market changed, not just the customers"],
      image="02_class_balance.png", img_w=7.2,
      takeaway="This is our best Kenyan proxy for a new market, and the model barely copes.")

slide("Model choice and the money",
      ["12 candidates on temporal folds; class weights never helped and broke calibration",
       "LogReg: best PR-AUC (0.143), best after Dec 2022; shallow LightGBM: best ROC",
       "Bootstrap: LogReg ≈ LightGBM on PR-AUC overall; LogReg significantly better after Dec 2022",
       "Per 1,000 applications after Dec 2022: approve-all −16,600 GHS; LogReg at 0.062 −3,500"],
      image="05_threshold_value.png", img_w=7.0,
      takeaway="The model earns its keep exactly where it is least accurate.")

slide("How much to lend",
      ["Ridge on log amount, trained on repaid loans only; median error ≈ 1,930 KES (GHS 150)",
       "Raw-scale terms extrapolated a 3-year loan to e^57 KES; log scale and a cap fixed it",
       "'Lend what they borrowed before' is within 25% more often than any model (38%)",
       "First-time borrowers over-predicted by ~40% → first-loan cap"],
      image="06_residuals.png", img_w=7.0)

slide("Telling a borrower why",
      ["Logistic log-odds split exactly into reason families; top 3 become plain-language reasons",
       "Caught: term and amount coefficients cancel, so a 'loan too big' reason pointed the wrong way",
       "Caught: Ghana applicants on unseen products would be told their product 'had problems'; now: 'new to us, no record yet'",
       "Calendar features moved scores without predicting anything → removed"],
      image="09_family_importance.png", img_w=6.4)

slide("Ghana: out of the model's reach",
      ["87% of loans on unseen products, 96% on unseen lenders, 66% co-funded",
       "Domain classifier: AUC 1.000 (0.999 without product, lender, co-funding)",
       "Importance weighting: effective sample 2.3% of Kenya; scores change, no way to check",
       "Model would decline 22% (Kenya 6.7%): almost all Type_2/Type_8, 8.5% of Type_3"],
      image="10_distributions.png", img_w=7.0,
      takeaway="No automated Ghana decisions until a capped pilot produces Ghanaian outcomes.")

slide("Fairness and the law",
      ["No protected attributes in the data, so we tested proxies",
       "Thin-file applicants: declined 32%, including 30% of those who would repay",
       "Dropping history features changes nothing (0.145 → 0.144): the proxy leaks back",
       "Findex Ghana: account gaps by gender, income, education, area; mobile-money borrowing reaches these groups",
       "Act 843 s.41: notice + 21-day human review; BoG: 20-day complaints, bureau reporting (not met)"],
      image="11_findex_gaps.png", img_w=6.6,
      takeaway="Mitigation is a policy: refer thin-file applicants to a capped starter loan, monitor parity.")

slide("Scoring service",
      ["Scorer.score(application, history) → P(default), decision, 3 reasons, offered amount",
       "Same feature code as training; tested to reproduce training features exactly",
       "approve · decline · refer (thin-file, or any market other than Kenya)",
       "~20 ms per call (p95 32 ms); 115,000 loans/s in batch",
       "Nine application fields + earlier loans; no personal or sensitive data"])

slide("What we recommend",
      ["Kenya: deploy logistic regression at the cost threshold, recalibrate monthly, monitor drift",
       "Thin-file applicants: starter loans and human review, never automatic decline",
       "Ghana: capped, monitored pilot with human review; collect outcomes, then recalibrate and validate",
       "Before launch: credit-bureau integration (BoG §7.3, Act 726), consent capture, legal review of Act 726 and Act 1052",
       "Replace our cost assumptions with NkwaPay's own"],
      takeaway="The next gain comes from Ghanaian data, not a cleverer Kenyan model.")

out = ROOT / "slides" / "NkwaPay_Debrief_Group3.pptx"
prs.save(out)
print(out)
for t, sz in FITS:
    print(f"{sz:>3} pt  {t}")
