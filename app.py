"""
Streamlit app - Santander Customer Satisfaction

Single-customer form that scores the probability of dissatisfaction using the
production model trained in Santander_Customer_Satisfaction.ipynb (Hist Gradient
Boosting, no rebalancing, Val ROC-AUC ~0.836 on the notebook's held-out fold).

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

BINARY_FEATURES = [
    "ind_var30", "ind_var5", "ind_var13_0", "ind_var12_0",
    "ind_var5_0", "ind_var12", "ind_var39_0",
]
COUNT_FEATURES = [
    "num_var30", "num_meses_var5_ult3", "num_var42", "num_var4",
    "num_var30_0", "num_var5_0", "num_var12_0", "num_var39_0",
]
OTHER_INT_FEATURES = ["var15", "var36", "n_zeros"]
MONEY_FEATURES = [
    "saldo_var30", "saldo_medio_var5_ult3", "saldo_medio_var5_hace2",
    "saldo_medio_var5_ult1", "saldo_var5", "saldo_var13",
    "saldo_medio_var5_hace3", "var38", "saldo_var12",
    "saldo_medio_var13_corto_hace2", "saldo_medio_var12_hace2",
]

FEATURE_HELP = {
    "n_zeros": "Quantos dos ~300 campos originais desse cliente valem zero (engenharia feita no notebook, Secao 6).",
    "var38_log": "Calculado automaticamente como log1p(var38) - nao e um input.",
    "var15": "Provavelmente relacionada a idade do cliente (o Kaggle nunca revelou o significado exato).",
    "var36": "Codigo categorico (canal/segmento) - valores tipicos: 0, 1, 2, 3 ou 99.",
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
        "Modelo: Hist Gradient Boosting (sem rebalanceamento) · "
        "treinado no dataset Kaggle Santander Customer Satisfaction · "
        "Val ROC-AUC ≈ 0.836 na avaliacao do notebook."
    )

    bundle = load_model()
    stats = load_stats()
    features = bundle["features"]

    with st.expander("Sobre este app", expanded=False):
        st.markdown(
            "Formulario para um unico cliente. As variaveis sao os nomes anonimizados "
            "originais do dataset do Kaggle (`varN`, `saldo_varN`, `num_varN`, `ind_varN`) - "
            "a competicao nunca revelou o significado real de cada uma, entao este app as "
            "apresenta como estao, agrupadas por tipo. O pipeline completo (limpeza, feature "
            "engineering, selecao de features e o modelo final) esta documentado em "
            "`Santander_Customer_Satisfaction.ipynb`."
        )

    st.markdown("### Dados do cliente")
    values = {}

    with st.form("customer_form"):
        st.markdown("**Indicadores (sim/nao)**")
        cols = st.columns(2)
        for i, feat in enumerate(BINARY_FEATURES):
            default = bool(round(stats.loc[feat, "50%"]))
            values[feat] = cols[i % 2].checkbox(feat, value=default, help=FEATURE_HELP.get(feat))

        st.markdown("**Contagens**")
        cols = st.columns(2)
        for i, feat in enumerate(COUNT_FEATURES):
            default = int(round(stats.loc[feat, "50%"]))
            values[feat] = cols[i % 2].number_input(
                feat, min_value=0, value=default, step=1, help=FEATURE_HELP.get(feat)
            )

        st.markdown("**Outros campos numericos**")
        cols = st.columns(2)
        for i, feat in enumerate(OTHER_INT_FEATURES):
            default = int(round(stats.loc[feat, "50%"]))
            lo = int(stats.loc[feat, "min"])
            values[feat] = cols[i % 2].number_input(
                feat, min_value=lo, value=default, step=1, help=FEATURE_HELP.get(feat)
            )

        st.markdown("**Saldos monetarios**")
        cols = st.columns(2)
        for i, feat in enumerate(MONEY_FEATURES):
            default = float(stats.loc[feat, "50%"])
            values[feat] = cols[i % 2].number_input(
                feat, value=default, step=1.0, format="%.2f", help=FEATURE_HELP.get(feat)
            )

        submitted = st.form_submit_button("Calcular probabilidade de insatisfacao", type="primary")

    if submitted:
        row = {feat: values[feat] for feat in features if feat != "var38_log"}
        row["var38_log"] = float(np.log1p(max(values["var38"], 0)))
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
