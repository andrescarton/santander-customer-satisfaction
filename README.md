# Santander Customer Satisfaction

End-to-end data science project on the [Kaggle Santander Customer Satisfaction](https://www.kaggle.com/c/santander-customer-satisfaction) dataset, plus a Streamlit app that serves the trained model.

**Author:** Andre Kim Scarton

## Problem

Santander wants to flag dissatisfied customers *before* they close their accounts. Unhappy
customers rarely complain first — they quietly leave. The dataset has 76,020 customers,
370 anonymized features, and a binary `TARGET` (1 = dissatisfied), with a strong class
imbalance (~4% dissatisfied).

## What's in this repo

- `Santander_Customer_Satisfaction.ipynb` — full analysis: data cleaning, leak-free feature
  engineering, feature selection (correlation + Mann-Whitney U + mutual information + Random
  Forest importance), training/comparison of 6 algorithms, significance testing, SMOTE vs.
  class-weighting ablation, precision-recall/business framing, model explanation, and the
  final Kaggle submission.
- `app.py` — Streamlit app that scores a single customer with the production model.
- `retrain_production_model.py` — reproduces the notebook's final pipeline (Sections 4, 6,
  7.6, 9.6, 12, 15) end to end and saves `models/best_model.pkl`. Re-run this if you upgrade
  numpy/scikit-learn and the pickle stops loading.
- `models/best_model.pkl` — the shipped model: Hist Gradient Boosting, no class rebalancing
  (chosen over SMOTE for much better-calibrated probabilities at equal ranking performance).
- `models/feature_stats.csv` — per-feature min/median/max used to pre-fill the app's form.

Raw data (`data/train.csv`, `data/test.csv`) and the five non-winning model files are not
included — download the dataset from Kaggle if you want to re-run the notebook from scratch.

## Results

| Model | Val ROC-AUC | CV ROC-AUC (mean ± std) |
|---|---|---|
| **Hist Gradient Boosting** | **0.8319** | 0.8211 ± 0.0037 |
| Random Forest | 0.8267 | 0.8216 ± 0.0028 |
| Decision Tree | 0.8166 | 0.8040 ± 0.0062 |
| SVM (RBF) | 0.7950 | 0.7990 ± 0.0227 |
| Logistic Regression | 0.7946 | 0.7902 ± 0.0032 |
| KNN | 0.7408 | 0.7204 ± 0.0041 |

Hist Gradient Boosting beats the runner-up (Random Forest) in 97.3% of 2,000 bootstrap
resamples. Final shipped configuration skips SMOTE in favor of no rebalancing: comparable
ranking (Val ROC-AUC 0.8364) with far better-calibrated probabilities (Brier score 0.0348 vs.
0.1309 with SMOTE).

## Running locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

If `models/best_model.pkl` fails to load (numpy/scikit-learn version mismatch), regenerate it
with `python retrain_production_model.py` (requires `data/train.csv` from Kaggle).

## Notes on the data

Feature names (`varN`, `saldo_varN`, `num_varN`, `ind_varN`) are anonymized by Kaggle — their
real-world meaning was never disclosed. The app presents them as-is, grouped by type
(indicators, counts, monetary balances), for whoever wants to try a hypothetical customer.
