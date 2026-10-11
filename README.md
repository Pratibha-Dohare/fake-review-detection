#  Fake Review Detection

A responsive Streamlit ML demo that flags **potentially** fake product reviews using TF-IDF, text-length signals, ratings, and optional reviewer metadata. Compares Logistic Regression and Random Forest with class weighting and a fake-class F1 score.


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
