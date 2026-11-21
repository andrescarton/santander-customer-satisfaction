# Santander Customer Satisfaction

**[Live app](https://santander-customer-satisfaction-mwi9wabakglouieoynhvj6.streamlit.app/)**

Predicting which bank customers are unhappy before they close their account, using the [Kaggle Santander Customer Satisfaction](https://www.kaggle.com/c/santander-customer-satisfaction) data. There's a small Streamlit app that scores a single customer with the final model.

Having worked in a bank, this problem felt familiar, so I wanted to do it properly: no leakage, honest validation, and a model small enough that someone could actually use it.

New to the project? [The story behind it](docs/PROJECT_STORY.md) walks through what I did and why, in plain language.

## The data

76,020 customers, 370 anonymised features and a binary target (1 = dissatisfied). Only about 4% are unhappy, so the classes are very imbalanced. Kaggle never said what the columns mean (`var15`, `saldo_var30`, ...), so the app gives them descriptive labels and keeps the original name next to each one.

## What I did

- Cleaned the data and engineered features without touching the validation set
- Selected features with correlation, Mann-Whitney U, mutual information and random forest importance
- Trained and compared 6 models, with bootstrap tests to check whether the differences were real
- Tested SMOTE against class weights and no rebalancing at all
- Framed the result with precision/recall for the business side
- Used permutation importance to cut the final model down from 30 features to 10

## Results

| Model | Val ROC-AUC | CV ROC-AUC (mean ± std) |
|---|---|---|
| **Hist Gradient Boosting** | **0.8319** | 0.8211 ± 0.0037 |
| Random Forest | 0.8267 | 0.8216 ± 0.0028 |
| Decision Tree | 0.8166 | 0.8040 ± 0.0062 |
| SVM (RBF) | 0.7950 | 0.7990 ± 0.0227 |
| Logistic Regression | 0.7946 | 0.7902 ± 0.0032 |
| KNN | 0.7408 | 0.7204 ± 0.0041 |

Hist Gradient Boosting beat Random Forest in 97.3% of 2,000 bootstrap resamples.

SMOTE didn't help. Without any rebalancing the ranking was about the same (Val ROC-AUC 0.8364), and the probabilities were far better calibrated: Brier score 0.0348, against 0.1309 with SMOTE. So the final model uses no rebalancing.

### 30 features down to 10

Most of the signal sits in two features, `var15` and `saldo_var30`. The bottom 20 or so are basically noise. Keeping only the top 10 costs almost nothing:

| Features | Val ROC-AUC | Change vs. 30 |
|---|---|---|
| 5 | 0.8324 | −0.0040 |
| 8 | 0.8343 | −0.0020 |
| **10** | **0.8361** | **−0.0002** |
| 15 | 0.8367 | +0.0004 |
| 30 | 0.8364 | – |

Ten fields is also a lot nicer to fill in on a form than thirty, which is what the app needed.

The 10 features: `var15`, `saldo_var30`, `saldo_medio_var5_hace3`, `var38_log`, `n_zeros`, `saldo_medio_var5_hace2`, `saldo_medio_var5_ult3`, `saldo_var5`, `num_var30_0`, `ind_var30`.

## Files

- `Santander_Customer_Satisfaction.ipynb` – the full analysis, including the Kaggle submission
- `app.py` – Streamlit app with a 10-field form
- `retrain_production_model.py` – rebuilds `models/best_model.pkl` from scratch
- `models/best_model.pkl` – the final model
- `models/feature_stats.csv` – min / median / max per feature, used to pre-fill the form
- `models/permutation_importance_30.csv` – importance of all 30 candidate features, if you want a different cutoff

The raw data isn't in the repo; download `train.csv` and `test.csv` from Kaggle into `data/`.

## Running it

```bash
pip install -r requirements.txt
streamlit run app.py
```

If the pickle won't load after a numpy or scikit-learn upgrade, run `python retrain_production_model.py` to rebuild it (needs `data/train.csv`).
