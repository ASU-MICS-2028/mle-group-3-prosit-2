"""Publish the web app to a Hugging Face Space.

    python app/deploy_space.py <username>/nkwapay-scorer [--private]

Copies only what the app needs into a staging folder, writes the Space's
README header, and uploads it. Run `hf auth login` once first.
"""
import argparse
import shutil
import tempfile
from pathlib import Path

from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parents[1]

SPACE_README = """---
title: NkwaPay Loan Scorer
emoji: 💳
colorFrom: red
colorTo: gray
sdk: streamlit
sdk_version: 1.45.1
python_version: "3.12"
app_file: app/app.py
pinned: false
short_description: Credit scoring demo, ICS553 Prosit 2 (Group 3)
---

Web front end for the NkwaPay credit-scoring model (ICS553 Machine Learning Essentials,
Prosit 2, Group 3). Logistic regression trained on public Kenyan loan data from the
African Credit Scoring Challenge. A class project, not a lending service.
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("space_id", help="e.g. yourname/nkwapay-scorer")
    ap.add_argument("--private", action="store_true")
    args = ap.parse_args()

    with tempfile.TemporaryDirectory() as tmp:
        stage = Path(tmp)
        (stage / "app").mkdir()
        shutil.copy(ROOT / "app" / "app.py", stage / "app" / "app.py")
        shutil.copytree(ROOT / "src", stage / "src", ignore=shutil.ignore_patterns("__pycache__", ".DS_Store"))
        shutil.copytree(ROOT / "examples", stage / "examples")
        (stage / "models").mkdir()
        shutil.copy(ROOT / "models" / "nkwapay-v1.joblib", stage / "models")
        shutil.copy(ROOT / "app" / "requirements.txt", stage / "requirements.txt")
        (stage / "README.md").write_text(SPACE_README)

        api = HfApi()
        api.create_repo(args.space_id, repo_type="space", space_sdk="streamlit",
                        private=args.private, exist_ok=True)
        api.upload_folder(folder_path=str(stage), repo_id=args.space_id, repo_type="space",
                          commit_message="Deploy NkwaPay scorer")
    print(f"Deployed: https://huggingface.co/spaces/{args.space_id}")


if __name__ == "__main__":
    main()
