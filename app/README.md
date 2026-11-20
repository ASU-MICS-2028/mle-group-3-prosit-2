# NkwaPay loan scorer (web app)

**Live:** https://nkwapay-scorer.streamlit.app

A small web front end for the shipped model (`models/nkwapay-v1.joblib`). Fill in an application and the applicant's earlier loans, or load one of the five examples, and the app shows the decision (approve, decline or refer), the chance of default against the 6.2% threshold, up to three plain-language reasons and the amount offered. It calls the same `Scorer` as `scripts/score.py`, so the answers are identical.

## Run it on your machine

```bash
pip install -r requirements.txt     # from the repository root
streamlit run app/app.py
```

## How it is deployed

The live app runs on Streamlit Community Cloud from [Xenongt1/nkwapay-scorer](https://github.com/Xenongt1/nkwapay-scorer), a copy of `app/`, `src/`, `models/` and `examples/`. It cannot deploy from this repository because the ASU-MICS-2028 organisation blocks third-party app access (an organisation owner would have to approve Streamlit).

To update the live app, copy the changed files into that repository and push; Streamlit redeploys within a minute or two. To deploy a new copy: sign in at https://share.streamlit.io with GitHub, Create app, choose the repository, branch `main`, main file `app/app.py`, Python 3.12 under Advanced settings, then Deploy.

Streamlit Cloud installs `app/requirements.txt` (the folder of the main file is checked first). Only the code, the saved model and the made-up examples are in the repository; no loan data is uploaded.

`deploy_space.py` publishes the same app to a Hugging Face Docker Space instead; Hugging Face now charges (PRO) for Docker Spaces.

The app is a class project on public competition data, not a lending service. Non-Kenyan applications are always referred, because the model is not validated outside Kenya.
