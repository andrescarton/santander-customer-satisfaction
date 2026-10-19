"""
Streamlit app - Santander Customer Satisfaction

Single-customer form that scores the probability of dissatisfaction using the
production model trained in Santander_Customer_Satisfaction.ipynb (Hist Gradient
Boosting, no rebalancing). The model uses only the top 10 of the notebook's 30
selected features, chosen by permutation importance in Section 16 - Val ROC-AUC
0.8361 vs. 0.8364 with all 30, so the smaller form costs essentially nothing in
accuracy.

Feature names are the original anonymized Kaggle variables (var15, saldo_var30, ...) -
the competition never revealed their real-world meaning, so this app presents them as-is,
grouped by type to stay usable.
"""
import numpy as np
import pandas as pd
import streamlit as st
import joblib

MODEL_PATH = "models/best_model.pkl"
STATS_PATH = "models/feature_stats.csv"

BINARY_FEATURES = ["ind_var30"]
COUNT_FEATURES = ["num_var30_0"]
OTHER_INT_FEATURES = ["var15", "n_zeros"]
MONEY_FEATURES = [
    "saldo_var30", "saldo_medio_var5_hace3", "saldo_medio_var5_hace2",
    "saldo_medio_var5_ult3", "saldo_var5",
]

FEATURE_LABELS = {
    "ind_var30": "ind_var30 (indicador)",
    "num_var30_0": "num_var30_0 (contagem)",
    "var15": "var15",
    "n_zeros": "n_zeros",
    "saldo_var30": "saldo_var30",
    "saldo_medio_var5_hace3": "saldo_medio_var5_hace3",
    "saldo_medio_var5_hace2": "saldo_medio_var5_hace2",
    "saldo_medio_var5_ult3": "saldo_medio_var5_ult3",
    "saldo_var5": "saldo_var5",
    "var38": "var38 (saldo/valor do cliente)",
}

FEATURE_HELP = {
    "n_zeros": "Quantos dos ~300 campos originais desse cliente valem zero (engenharia feita no notebook, Secao 6).",
    "var15": "Provavelmente relacionada a idade do cliente (o Kaggle nunca revelou o significado exato). A feature mais importante do modelo.",
    "var38": "Usado para calcular var38_log = log1p(var38), a 4a feature mais importante. var38 em si nao entra direto no modelo.",
}


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_stats():
    return pd.read_csv(STATS_PATH, index_col=0)


def risk_tier(prob):
    if prob < 0.10:
        return "Baixo", "🟢"
    if prob < 0.30:
        return "Medio", "🟡"
    return "Alto", "🔴"


def main():
    st.set_page_config(page_title="Santander Customer Satisfaction", page_icon="🏦", layout="centered")
    st.title("🏦 Previsao de insatisfacao de clientes")
    st.caption(
        "Modelo: Hist Gradient Boosting (sem rebalanceamento), 10 features · "
        "treinado no dataset Kaggle Santander Customer Satisfaction · "
        "Val ROC-AUC ≈ 0.836."
    )

    bundle = load_model()
    stats = load_stats()
    features = bundle["features"]

    with st.expander("Sobre este app", expanded=False):
        st.markdown(
            "Formulario para um unico cliente. O modelo original usava 30 variaveis; um teste de "
            "permutation importance (notebook, Secao 16) mostrou que **10 delas** respondem por "
            "praticamente todo o poder preditivo (AUC cai so 0.0002), entao o app usa so essas 10. "
            "Os nomes sao os originais e anonimizados do Kaggle - a competicao nunca revelou o "
            "significado real de cada variavel."
        )

    st.markdown("### Dados do cliente")
    values = {}

    with st.form("customer_form"):
        st.markdown("**Indicador e contagem**")
        cols = st.columns(2)
        for i, feat in enumerate(BINARY_FEATURES):
            default = bool(round(stats.loc[feat, "50%"]))
            values[feat] = cols[i % 2].checkbox(FEATURE_LABELS[feat], value=default, help=FEATURE_HELP.get(feat))
        for i, feat in enumerate(COUNT_FEATURES):
            default = int(round(stats.loc[feat, "50%"]))
            values[feat] = cols[(i + 1) % 2].number_input(
                FEATURE_LABELS[feat], min_value=0, value=default, step=1, help=FEATURE_HELP.get(feat)
            )

        st.markdown("**Outros campos numericos**")
        cols = st.columns(2)
        for i, feat in enumerate(OTHER_INT_FEATURES):
            default = int(round(stats.loc[feat, "50%"]))
            lo = int(stats.loc[feat, "min"])
            values[feat] = cols[i % 2].number_input(
                FEATURE_LABELS[feat], min_value=lo, value=default, step=1, help=FEATURE_HELP.get(feat)
            )

        st.markdown("**Saldos monetarios**")
        cols = st.columns(2)
        for i, feat in enumerate(MONEY_FEATURES):
            default = float(stats.loc[feat, "50%"])
            values[feat] = cols[i % 2].number_input(
                FEATURE_LABELS[feat], value=default, step=1.0, format="%.2f", help=FEATURE_HELP.get(feat)
            )
        # var38 isn't a model feature itself - only var38_log (derived from it) is. Its default
        # is back-computed from the var38_log stats so the form doesn't need a separate stats row.
        var38_default = float(np.expm1(stats.loc["var38_log", "50%"]))
        values["var38"] = cols[len(MONEY_FEATURES) % 2].number_input(
            FEATURE_LABELS["var38"], min_value=0.0, value=var38_default, step=1.0, format="%.2f",
            help=FEATURE_HELP.get("var38"),
        )

        submitted = st.form_submit_button("Calcular probabilidade de insatisfacao", type="primary")

    if submitted:
        row = {feat: values[feat] for feat in features if feat != "var38_log"}
        row["var38_log"] = float(np.log1p(values["var38"]))
        for feat in BINARY_FEATURES:
            row[feat] = int(row[feat])

        X = pd.DataFrame([row])[features]
        prob = float(bundle["production_model"].predict_proba(X)[0, 1])
        tier, emoji = risk_tier(prob)

        st.markdown("### Resultado")
        c1, c2 = st.columns(2)
        c1.metric("Probabilidade de insatisfacao", f"{prob:.1%}")
        c2.metric("Nivel de risco", f"{emoji} {tier}")
        st.progress(min(prob, 1.0))
        st.caption(
            "Base rate historica do dataset: ~4% dos clientes sao insatisfeitos (TARGET=1). "
            "Nivel de risco: Baixo < 10%, Medio 10-30%, Alto > 30% - limiares ilustrativos, "
            "nao calibrados com custo de negocio real (ver Secao 13 do notebook)."
        )


if __name__ == "__main__":
    main()
