# NkwaPay loan scorer (web app)

A small web front end for the shipped model (`models/nkwapay-v1.joblib`). Fill in an application and the applicant's earlier loans, or load one of the five examples, and the app shows the decision (approve, decline or refer), the chance of default against the 6.2% threshold, up to three plain-language reasons and the amount offered. It calls the same `Scorer` as `scripts/score.py`, so the answers are identical.

## Run it on your machine

```bash
pip install -r requirements.txt     # from the repository root
streamlit run app/app.py
```

## Put it online (Streamlit Community Cloud, free)

1. Sign in at https://share.streamlit.io with GitHub.
2. Create app → "Deploy a public app from GitHub".
3. Repository `ASU-MICS-2028/mle-group-3-prosit-2`, branch `app/streamlit-scorer` (or `main` once merged), main file path `app/app.py`.
4. Under Advanced settings choose Python 3.12, then Deploy.

Streamlit Cloud installs `app/requirements.txt` (the folder of the main file is checked first). Only the code, the saved model and the made-up examples are in the repository; no loan data is uploaded.

`deploy_space.py` publishes the same app to a Hugging Face Docker Space instead; Hugging Face now charges (PRO) for Docker Spaces.

The app is a class project on public competition data, not a lending service. Non-Kenyan applications are always referred, because the model is not validated outside Kenya.
