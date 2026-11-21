# The story behind this project

The [README](../README.md) has the numbers. This page explains what I did and why, in plain language, for anyone who wants the full picture without reading a 110-cell notebook.

## The question

Banks lose customers quietly. An unhappy customer rarely calls to complain. They just stop using the account and, one day, close it. If you could spot the warning signs a few months early, someone could pick up the phone, fix the problem and keep the customer.

Santander put this question to Kaggle in 2016: given what the bank knows about a customer, can you tell whether they are unhappy?

I spent five years at a bank in Brazil, so this one felt close to home.

## The data, and what was odd about it

The dataset has **76,020 customers** and **370 columns**, plus one answer column: `1` if the customer was dissatisfied, `0` if not.

Three things stood out right away.

**Every column name is a code.** Santander anonymised the data, so instead of "age" or "account balance" you get names like `var15`, `saldo_var30` and `num_var30_0`. Some hints survive, because the names are partly Spanish (`saldo` is balance, `hace3` is "three months ago", `ult3` is "last three months"), but nobody outside the bank knows exactly what each column means. That shaped the whole project: I couldn't lean on banking intuition, so every decision had to come from the data itself.

**Almost everyone is happy.** Only 3,008 customers, about **4%**, are dissatisfied. That sounds like good news, but it makes the problem hard. A model that answers "happy" for everyone is right 96% of the time and completely useless, because it never finds a single unhappy customer. So I ignored accuracy and measured **ROC-AUC** instead. In simple terms, it asks: if you pick one unhappy and one happy customer at random, how often does the model rank the unhappy one as riskier? 0.5 is a coin flip, 1.0 is perfect.

**A lot of the columns were junk.** 34 columns had the same value for every customer, and 62 were exact copies of another column. After removing them, 308 columns were left. Two more problems were hiding in the data:

- `var3` was `2` for almost everyone, except 116 customers with the value **-999,999**. Nobody has -999,999 of anything. That's a placeholder someone used instead of leaving the cell empty, so I treated it as missing.
- `var38` (probably some kind of total value with the bank) was extremely lopsided. Most customers sat around 100,000, while a few went past 22 million. Models struggle with that shape, so I used its logarithm instead, which pulls the extremes in.

## The rule I set early: don't peek at the exam

Before fixing anything, I put a quarter of the customers aside as a **validation set** and didn't look at them again until it was time to grade the models.

This sounds like a small detail, but it's where a lot of projects quietly go wrong. If you fill in missing values, pick features or tune a model using the whole dataset, information from the "exam" leaks into the "study material", and the final score looks better than it will be in real life. So every number I learned (the value used to fill `var3`, which features to keep, the model settings) came from the training side only. The validation score in this project is a fair estimate of how the model would do on customers it has never seen.

## Two features I made myself

- **`var38_log`**, the log version of `var38` described above.
- **`n_zeros`**, a count of how many of a customer's ~300 fields are zero. Most columns describe products and transactions, so a customer with lots of zeros barely uses the bank. My guess was that disengaged customers are more likely to be unhappy, and the model later agreed: `n_zeros` ended up as the 5th most useful input.

## Picking the columns that matter

300 columns is too many to throw at a model, and most of them carry no useful signal. Instead of trusting a single method, I ran four different ones and let them vote:

1. How strongly each column moves with the answer (correlation)
2. Whether happy and unhappy customers have clearly different values (a Mann-Whitney U test)
3. How much knowing the column reduces uncertainty about the answer (mutual information)
4. How much a Random Forest relies on it

Columns that ranked well across all four made the cut. That gave me 30.

## Six models, one winner

I trained six different kinds of model on those 30 columns and compared them on the validation set:

| Model | Validation ROC-AUC |
|---|---|
| **Histogram Gradient Boosting** | **0.832** |
| Random Forest | 0.827 |
| Decision Tree | 0.817 |
| SVM | 0.795 |
| Logistic Regression | 0.795 |
| K-Nearest Neighbours | 0.741 |

The top two are close, so I checked whether the gap was real or just luck. I resampled the validation set 2,000 times and recomputed both scores each time. Gradient Boosting came out ahead in **97.3%** of them. It's not a landslide, but it's consistent enough to choose it.

## The SMOTE surprise

With only 4% unhappy customers, the textbook move is to rebalance the data. SMOTE, the most popular technique, creates synthetic unhappy customers until the two groups are the same size. I used it at first, and then tested whether it was actually helping.

It wasn't. Without any rebalancing the model ranked customers just as well (ROC-AUC 0.836 against 0.832), and its probabilities were far more trustworthy.

What does "trustworthy" mean here? If the model says a group of customers has a 10% chance of being unhappy, about 10 in 100 of them should actually be unhappy. A model trained on SMOTE's 50/50 world forgets that unhappy customers are rare, so it overestimates everyone's risk. The Brier score, which measures exactly this kind of error (lower is better), was **0.131 with SMOTE and 0.035 without it**. So the final model uses no rebalancing.

## What it means for the bank

A score like "ROC-AUC 0.83" doesn't tell a retention team what to do on Monday morning, so I turned it into a simpler question: if the team can only call the top X% riskiest customers, what do they get?

| Customers called | Share of all unhappy customers found | Calls that reach an unhappy customer |
|---|---|---|
| Top 1% | 7% | 26% |
| Top 5% | 26% | 21% |
| Top 10% | 49% | 19% |
| Top 20% | 69% | 14% |

Calling the top 10% finds about **half of all unhappy customers**, and roughly **1 in 5 calls** reaches someone who is actually unhappy. Calling at random, it would be 1 in 25. Where to draw the line depends on what a call costs and what a lost customer costs, and that's a business decision, not a modelling one.

## What drives the predictions

To see what the model relies on, I shuffled one column at a time and measured how much worse the predictions got. If shuffling a column hurts a lot, the model needs it.

Two columns dominate:

- **`var15`**, almost certainly the customer's **age**. Shuffling it drops ROC-AUC by 0.17, which is huge.
- **`saldo_var30`**, the balance on one type of account. Shuffling it drops ROC-AUC by 0.09.

After those two, the impact falls off fast, and about 20 of the 30 columns barely mattered at all.

## Cutting 30 inputs down to 10

That last finding made me wonder how many columns the model really needed. I retrained it with just the top 5, 8, 10, 15 and all 30:

| Columns kept | Validation ROC-AUC |
|---|---|
| 5 | 0.832 |
| 8 | 0.834 |
| **10** | **0.836** |
| 15 | 0.837 |
| 30 | 0.836 |

With 10 columns the score is practically identical to 30. A smaller model is easier to explain, cheaper to run and, for the app, a form with 10 fields instead of 30. That's the version I shipped.

## The app

The [live app](https://santander-customer-satisfaction-mwi9wabakglouieoynhvj6.streamlit.app/) lets you type in a customer's details and see their estimated risk. Every field starts at the value of a typical customer from the data, so you can change one thing at a time and watch the number move. Try changing the age or the account 30 balance, the two inputs the model cares about most.

Because the real column meanings are unknown, the labels in the app ("Age", "Relationship value", "Current balance") are my best interpretation. The original column name sits next to each one so nothing is hidden.

## Things that went wrong along the way

Not everything was smooth, and these were useful lessons:

- **The saved model stopped loading.** After a routine library upgrade, the saved model file wouldn't open anymore. Saved Python models are tied to the library versions they were built with. I wrote `retrain_production_model.py` so the model can be rebuilt from scratch in one command.
- **The first deploys failed.** Streamlit Cloud runs a newer Python than my laptop, and one of the library versions I'd pinned had no ready-made build for it, so the server tried to compile it from source and hung. The fix was to pick versions that work on both and test the install in a clean environment that matches the server before pushing.

## Limits worth knowing

- The data is from 2016 and the columns are anonymised, so this is a modelling exercise, not something to plug into a real bank as it is.
- The Low / Medium / High risk bands in the app (below 10%, 10–30%, above 30%) are illustrative. Real thresholds should come from real costs.
- Validation ROC-AUC of 0.836 is solid for this dataset, but the model still misses unhappy customers who look just like happy ones on paper.

## What I'd try next

- XGBoost, LightGBM or CatBoost, which often edge out scikit-learn's gradient boosting
- A smarter hyperparameter search with Optuna
- Combining the top models instead of picking one
- Setting the call threshold from actual business costs

---

The full analysis, with every chart and number, is in [`Santander_Customer_Satisfaction.ipynb`](../Santander_Customer_Satisfaction.ipynb).
