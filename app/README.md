# NkwaPay loan scorer (web app)

A small web front end for the shipped model (`models/nkwapay-v1.joblib`). Fill in an application and the applicant's earlier loans, or load one of the five examples, and the app shows the decision (approve, decline or refer), the chance of default against the 6.2% threshold, up to three plain-language reasons and the amount offered. It calls the same `Scorer` as `scripts/score.py`, so the answers are identical.

## Run it on your machine

```bash
pip install -r requirements.txt     # from the repository root
streamlit run app/app.py
```

## Put it online (Hugging Face Spaces)

```bash
pip install huggingface_hub
hf auth login                       # once, with a write token from huggingface.co/settings/tokens
python app/deploy_space.py <your-hf-username>/nkwapay-scorer
```

`deploy_space.py` copies only what the app needs (`app/`, `src/`, `models/`, `examples/`) into a Space and pushes it. No loan data is uploaded: the saved model holds coefficients and averages only.

The app is a class project on public competition data, not a lending service. Non-Kenyan applications are always referred, because the model is not validated outside Kenya.
