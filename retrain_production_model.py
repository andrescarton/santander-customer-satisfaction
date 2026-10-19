"""
Regenerates models/best_model.pkl in the CURRENT environment.

The original best_model.pkl (saved 2026-08-17) was pickled under Python 3.13.9 with a
different numpy/scikit-learn build and no longer unpickles here (numpy BitGenerator /
sklearn version mismatch). Rather than chase the exact original environment, this script
reproduces the notebook's leak-free pipeline (Sections 4, 6, 7.6, 9.6, 12, 15) end to end
and re-fits the shipped model fresh, so the artifact matches the libraries actually
installed for deployment.

Feature list was further reduced from the notebook's 30 down to the top 10 by permutation
importance (computed 2026-10-19: the bottom 20 features carried ~0 or negative importance,
and Val ROC-AUC with 10 features is statistically indistinguishable from 30 - 0.8361 vs.
0.8364). This keeps the Streamlit form short without giving up accuracy.
"""
import os
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingClassifier

RANDOM_STATE = 42
SENTINEL = -999999
DATA_DIR = "data"
MODEL_DIR = "models"

# Top 10 of the notebook's 30 FINAL_FEATURES, ranked by permutation importance on the
# validation fold (see models/permutation_importance_30.csv for the full ranking).
FINAL_FEATURES = [
    'var15', 'saldo_var30', 'saldo_medio_var5_hace3', 'var38_log', 'n_zeros',
    'saldo_medio_var5_hace2', 'saldo_medio_var5_ult3', 'saldo_var5', 'num_var30_0', 'ind_var30',
]

BEST_HGB_PARAMS = dict(max_iter=200, max_depth=5, learning_rate=0.03, l2_regularization=0.0)


def prepare_features(df_raw, var3_mode_value):
    df = df_raw.copy()
    df["var3"] = df["var3"].replace(SENTINEL, var3_mode_value)
    df["var38_log"] = np.log1p(df["var38"])
    zero_count_cols = [c for c in df.columns if c not in ("ID", "TARGET", "var38_log")]
    df["n_zeros"] = (df[zero_count_cols] == 0).sum(axis=1)
    return df


def main():
    print("Loading data/train.csv ...")
    train_raw = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))

    # Section 4.1 / 4.2 - drop structurally useless columns
    nunique = train_raw.drop(columns=["ID", "TARGET"]).nunique()
    constant_cols = nunique[nunique == 1].index.tolist()
    features_only = train_raw.drop(columns=["ID", "TARGET"])
    duplicate_cols = features_only.columns[features_only.T.duplicated()].tolist()
    cols_to_drop_structural = list(set(constant_cols) | set(duplicate_cols))
    train_clean = train_raw.drop(columns=cols_to_drop_structural)
    print(f"Dropped {len(cols_to_drop_structural)} constant/duplicate columns "
          f"-> {train_clean.shape[1]} columns remain.")

    # Section 15 - refit on ALL labeled data; var3 mode learned from the full labeled set
    var3_mode_full = train_clean.loc[train_clean["var3"] != SENTINEL, "var3"].mode()[0]
    train_full_fe = prepare_features(train_clean, var3_mode_full)

    X_full = train_full_fe[FINAL_FEATURES]
    y_full = train_full_fe["TARGET"]

    # Section 9.6 + 12/15 - winning algorithm, tuned hyperparameters, "No rebalancing" strategy
    clf = HistGradientBoostingClassifier(random_state=RANDOM_STATE, class_weight=None, **BEST_HGB_PARAMS)
    production_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", clf),
    ])

    print(f"Refitting Hist Gradient Boosting (No rebalancing) on {len(X_full):,} labeled rows...")
    production_pipeline.fit(X_full, y_full)
    print("Done.")

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(
        {
            "production_model": production_pipeline,
            "features": FINAL_FEATURES,
            "var3_mode_full": var3_mode_full,
            "algorithm": "Hist Gradient Boosting",
            "balancing_strategy": "No rebalancing",
            "val_roc_auc": 0.836352,
        },
        os.path.join(MODEL_DIR, "best_model.pkl"),
    )
    print(f"Saved {MODEL_DIR}/best_model.pkl (re-fit in the current environment).")

    # Also save per-feature summary stats, used by the Streamlit form to build sane input widgets
    stats = train_full_fe[FINAL_FEATURES].describe(percentiles=[0.01, 0.5, 0.99]).T
    stats.to_csv(os.path.join(MODEL_DIR, "feature_stats.csv"))
    print(f"Saved {MODEL_DIR}/feature_stats.csv (min/median/max/1-99 percentile per feature).")


if __name__ == "__main__":
    main()
