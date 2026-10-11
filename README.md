# 🛡️ ReviewGuard — Fake Review Detection

A responsive Streamlit ML demo that flags **potentially** fake product reviews using TF-IDF, text-length signals, ratings, and optional reviewer metadata. Compares Logistic Regression and Random Forest with class weighting and a fake-class F1 score.

> ⚠️ The included labeled reviews are synthetic examples, NOT a benchmark. A flagged review is not evidence of actual fraud. Replace the demo CSV for meaningful research, and validate on independent real-world data.

## Run locally

```bash
python -m pip install -r requirements.txt
streamlit run app.py
```

Optional command-line training (saves `review_model.joblib` locally):

```bash
python model.py
```

The web app trains a small demo model in memory on first use. An uploaded CSV can replace it for the current user session.

## Deploy on Streamlit Community Cloud

1. Create a **public GitHub repository** and upload all the project files, including the `data/` and `.streamlit/` folders.
2. Visit https://share.streamlit.io/ and sign in with GitHub.
3. Click **Create app** → **Deploy a public app from GitHub**.
4. Select your repository, branch `main`, and the entrypoint `app.py`.
5. Deploy; Streamlit installs `requirements.txt` automatically.

**GitHub Pages cannot host Streamlit/Python apps by itself.** It can host static frontends; use Streamlit Community Cloud for this app.

## CSV format

At minimum:

```csv
review,label
"The device arrived on time and works as described.",genuine
"Absolutely best product ever!! Buy now!!!",fake
```

Labels: `fake`/`genuine`, `real`, or `1`/`0` (where 1 means fake). Add these optional numeric columns when available:

- `rating`: integer 1–5
- `verified_purchase`: `yes`/`no`, `true`/`false`, or 1/0
- `reviewer_review_count`: integer count

## ML pipeline

- Text → TF-IDF unigram + bigram vectors (learned on training data only).
- Extra features → review length, exclamation count, rating, verified purchase, reviewer activity count.
- Missing optional metadata → zero imputation, then scaled.
- Stratified 75/25 split; class-balanced Logistic Regression and Random Forest.
- Choose higher fake-class F1 on held-out split; show precision, recall, F1, and confusion matrix.

For rigorous published results, use train/validation/test or nested CV, calibrate probabilities, avoid leakage by grouping reviewers/products, and only evaluate on the test set **once** after choosing the model. The simple demo picks its model using the test set, so its displayed performance is optimistic.

## Project structure

```text
fake-review-detector/
├── app.py
├── model.py
├── data/
│   └── reviews.csv
├── .streamlit/
│   └── config.toml
├── requirements.txt
├── .gitignore
└── README.md
```
