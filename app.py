"""
Streamlit app - Santander Customer Satisfaction

Single-customer form that scores the probability of dissatisfaction using the
production model trained in Santander_Customer_Satisfaction.ipynb (Hist Gradient
Boosting, no rebalancing). The model uses only the top 10 of the notebook's 30
selected features, chosen by permutation importance in Section 16 - Val ROC-AUC
0.8361 vs. 0.8364 with all 30, so the smaller form costs essentially nothing in
accuracy.

The Kaggle competition never revealed what the anonymized variables mean, so the
form groups them by the account they refer to (var5, var30) and gives each a
descriptive label, keeping the original column name visible next to it.
"""
import html

import joblib
import numpy as np
import pandas as pd
import streamlit as st

MODEL_PATH = "models/best_model.pkl"
STATS_PATH = "models/feature_stats.csv"
REPO_URL = "https://github.com/andrescarton/santander-customer-satisfaction"

BASE_RATE = 0.0396  # share of TARGET=1 in the 76,020 labeled training customers
LOW_MAX, MEDIUM_MAX = 0.10, 0.30  # illustrative risk bands, not tuned to business costs
SCALE_MAX = 0.50  # the result bar spans 0-50%; anything above is pinned to the end

# (feature, label, help) per form section. var38 is not a model feature itself - the
# model uses var38_log = log1p(var38) - but it's the value a person would actually enter.
SECTIONS = {
    "Customer profile": [
        ("var15", "Age", "Anonymized as var15; widely read as the customer's age. The model's most important input."),
        ("var38", "Relationship value", "Anonymized as var38; often read as the customer's total value with the bank. The model uses log(1 + var38)."),
        ("n_zeros", "Empty fields", "How many of the ~300 original fields are zero for this customer - a proxy for how little of the bank's product range they use."),
    ],
    "Account type 30": [
        ("ind_var30", "Holds this account", None),
        ("num_var30_0", "Number of products", None),
        ("saldo_var30", "Current balance", None),
    ],
    "Account type 5": [
        ("saldo_var5", "Current balance", None),
        ("saldo_medio_var5_ult3", "Avg. balance, last 3 months", None),
        ("saldo_medio_var5_hace2", "Avg. balance, 2 months ago", None),
        ("saldo_medio_var5_hace3", "Avg. balance, 3 months ago", None),
    ],
}
BINARY_FEATURES = {"ind_var30"}
INT_FEATURES = {"var15", "n_zeros", "num_var30_0"}

CSS = """
<style>
  .block-container { max-width: 760px; padding-top: 2.5rem; }
  h1 { font-weight: 650; letter-spacing: -0.02em; margin-bottom: 0.25rem; }
  .eyebrow { font-size: 0.78rem; letter-spacing: 0.08em; text-transform: uppercase;
             color: #1F4E79; font-weight: 600; margin-bottom: 0.25rem; }
  .lede { color: #555; font-size: 1rem; line-height: 1.55; margin-bottom: 1.5rem; }
  .section-title { font-size: 0.95rem; font-weight: 600; margin: 0.25rem 0 0.1rem; }
  @media (max-width: 640px) { h1 { font-size: 2rem !important; } }
  [data-testid="stWidgetLabel"] code { color: #8A8A84; background: transparent; padding: 0; font-size: 0.72rem; }

  .result { border: 1px solid #DDDDD6; border-radius: 10px; background: #FFFFFF; padding: 1.5rem 1.6rem 1.3rem; margin-top: 0.5rem; }
  .result-label { font-size: 0.85rem; color: #666; }
  .result-row { display: flex; align-items: baseline; gap: 0.9rem; flex-wrap: wrap; margin: 0.2rem 0 1.2rem; }
  .result-value { font-size: 2.6rem; font-weight: 650; letter-spacing: -0.02em; line-height: 1.1; }
  .pill { font-size: 0.8rem; font-weight: 600; padding: 0.2rem 0.6rem; border-radius: 999px; }
  .pill.low { background: #E3F1E8; color: #1E6B3A; }
  .pill.medium { background: #FBF0D9; color: #8A5A00; }
  .pill.high { background: #FBE3E1; color: #A1261B; }
  .relative { font-size: 0.9rem; color: #555; }

  .bar { position: relative; height: 10px; border-radius: 5px; display: flex; overflow: hidden; }
  .bar span { height: 100%; }
  .marker { position: absolute; top: -5px; width: 3px; height: 20px; background: #1C1C1C; border-radius: 2px; transform: translateX(-50%); }
  .bar-wrap { position: relative; margin: 0 0 0.4rem; }
  .base-tick { position: absolute; top: -3px; height: 16px; border-left: 2px dotted #666; }
  .scale { position: relative; height: 1.1rem; font-size: 0.72rem; color: #888; }
  .scale span { position: absolute; transform: translateX(-50%); }
  .scale span:first-child { transform: none; }
  .scale span:last-child { transform: translateX(-100%); }
  .legend { font-size: 0.78rem; color: #777; margin-top: 0.6rem; }

  .footer { font-size: 0.78rem; color: #888; margin-top: 2.5rem; border-top: 1px solid #E4E4DE; padding-top: 0.9rem; line-height: 1.6; }
  .footer a { color: #1F4E79; }
</style>
"""


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_stats():
    return pd.read_csv(STATS_PATH, index_col=0)


def risk_tier(prob):
    if prob < LOW_MAX:
        return "Low risk", "low"
    if prob < MEDIUM_MAX:
        return "Medium risk", "medium"
    return "High risk", "high"


def field_label(label, feat):
    return f"{label}  ·  `{feat}`"


def feature_input(container, feat, label, help_text, stats):
    if feat == "var38":
        # Back-computed from the var38_log stats so no separate stats row is needed.
        default = float(np.expm1(stats.loc["var38_log", "50%"]))
        return container.number_input(field_label(label, feat), min_value=0.0, value=round(default, 2),
                                      step=1000.0, format="%.2f", help=help_text)
    median = stats.loc[feat, "50%"]
    if feat in BINARY_FEATURES:
        return container.selectbox(field_label(label, feat), ["Yes", "No"], index=0 if round(median) else 1,
                                   help=help_text) == "Yes"
    if feat in INT_FEATURES:
        return container.number_input(field_label(label, feat), min_value=int(stats.loc[feat, "min"]),
                                      value=int(round(median)), step=1, help=help_text)
    return container.number_input(field_label(label, feat), value=float(median), step=100.0,
                                  format="%.2f", help=help_text)


def pct(x):
    return f"{x * 100:g}%"


def result_card(prob):
    tier, css_class = risk_tier(prob)
    ratio = prob / BASE_RATE
    relative = f"{ratio:.1f}× the average customer ({BASE_RATE:.1%})"

    def pos(x):
        return min(x, SCALE_MAX) / SCALE_MAX * 100

    bands = (
        f'<span style="width:{pos(LOW_MAX)}%;background:#CFE6D7"></span>'
        f'<span style="width:{pos(MEDIUM_MAX) - pos(LOW_MAX)}%;background:#F3DFB0"></span>'
        f'<span style="width:{100 - pos(MEDIUM_MAX)}%;background:#F0C4BF"></span>'
    )
    ticks = [0, LOW_MAX, MEDIUM_MAX, SCALE_MAX]
    scale = "".join(f'<span style="left:{pos(t)}%">{pct(t)}{"+" if t == SCALE_MAX else ""}</span>' for t in ticks)

    return f"""
    <div class="result">
      <div class="result-label">Estimated probability of dissatisfaction</div>
      <div class="result-row">
        <div class="result-value">{prob:.1%}</div>
        <span class="pill {css_class}">{html.escape(tier)}</span>
        <span class="relative">{relative}</span>
      </div>
      <div class="bar-wrap">
        <div class="bar">{bands}</div>
        <div class="base-tick" style="left:{pos(BASE_RATE)}%"></div>
        <div class="marker" style="left:{pos(prob)}%"></div>
      </div>
      <div class="scale">{scale}</div>
      <div class="legend">Solid marker: this customer. Dotted line: dataset average.
        Risk bands are illustrative and not tuned to real business costs.</div>
    </div>
    """


def main():
    st.set_page_config(page_title="Customer Dissatisfaction Risk", layout="centered")
    st.markdown(CSS, unsafe_allow_html=True)

    st.markdown('<div class="eyebrow">Santander Customer Satisfaction · Kaggle</div>', unsafe_allow_html=True)
    st.title("Customer dissatisfaction risk")
    st.markdown(
        '<p class="lede">Enter a customer\'s banking profile to estimate how likely they are to be '
        "dissatisfied. Fields start at the median customer from the training data.</p>",
        unsafe_allow_html=True,
    )

    bundle = load_model()
    stats = load_stats()
    features = bundle["features"]

    values = {}
    with st.form("customer_form", border=False):
        for title, fields in SECTIONS.items():
            with st.container(border=True):
                st.markdown(f'<div class="section-title">{title}</div>', unsafe_allow_html=True)
                cols = st.columns(2)
                for i, (feat, label, help_text) in enumerate(fields):
                    values[feat] = feature_input(cols[i % 2], feat, label, help_text, stats)

        submitted = st.form_submit_button("Estimate risk", type="primary", width="stretch")

    if submitted:
        row = {feat: values[feat] for feat in features if feat != "var38_log"}
        row["var38_log"] = float(np.log1p(values["var38"]))
        for feat in BINARY_FEATURES:
            row[feat] = int(row[feat])

        X = pd.DataFrame([row])[features]
        prob = float(bundle["production_model"].predict_proba(X)[0, 1])
        st.markdown(result_card(prob), unsafe_allow_html=True)

    st.markdown(
        f"""<div class="footer">
        Model: Histogram Gradient Boosting on the 10 most important of 30 selected features
        (validation ROC-AUC 0.836), trained on 76,020 customers from the Kaggle
        Santander Customer Satisfaction dataset. Variable names are anonymized in the original data,
        so labels are descriptive interpretations.
        <a href="{REPO_URL}" target="_blank">Notebook and code on GitHub</a>.
        </div>""",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
