import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Passos Mágicos — Risco de Defasagem", layout="centered")

REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = REPO_ROOT / "models" / "q9_model.pkl"
META_PATH = REPO_ROOT / "models" / "q9_metadata.json"


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Modelo não encontrado: {MODEL_PATH}")
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_metadata():
    if META_PATH.exists():
        return json.loads(META_PATH.read_text(encoding="utf-8"))
    return {}


model = load_model()
meta = load_metadata()

st.title("Passos Mágicos — Preditor de Risco de Defasagem")
st.caption("Probabilidade de risco (Defas no top quartil) com base em indicadores e contexto.")

with st.expander("Detalhes do modelo"):
    st.json(meta)

st.subheader("Entradas do aluno")

# Numéricas
ida = st.number_input("IDA (Desempenho acadêmico)", min_value=0.0, max_value=10.0, value=6.0, step=0.1)
ieg = st.number_input("IEG (Engajamento)", min_value=0.0, max_value=10.0, value=7.5, step=0.1)
iaa = st.number_input("IAA (Autoavaliação)", min_value=0.0, max_value=10.0, value=8.0, step=0.1)
ips = st.number_input("IPS (Psicossocial)", min_value=0.0, max_value=10.0, value=7.0, step=0.1)

ano_ingresso = st.number_input("Ano de ingresso", min_value=2000, max_value=2035, value=2022, step=1)
idade_22 = st.number_input("Idade (em 2022)", min_value=5, max_value=30, value=12, step=1)

# Categóricas (valores podem variar — OneHotEncoder handle_unknown segura)
fase = st.selectbox("Fase", options=list(range(0, 8)), index=0)
pedra = st.selectbox("Pedra 22", options=["Quartzo", "Ágata", "Ametista", "Topázio"], index=0)
genero = st.selectbox("Gênero", options=["F", "M", "NR"], index=0)
turma = st.text_input("Turma", value="T1")

input_df = pd.DataFrame([{
    "IDA": ida,
    "IEG": ieg,
    "IAA": iaa,
    "IPS": ips,
    "Ano ingresso": ano_ingresso,
    "Idade 22": idade_22,
    "Fase": fase,
    "Pedra 22": pedra,
    "Gênero": genero,
    "Turma": turma,
}])

st.divider()

if st.button("Calcular risco", type="primary"):
    proba = float(model.predict_proba(input_df)[:, 1][0])
    st.metric("Probabilidade de risco", f"{proba*100:.1f}%")

    if proba >= 0.70:
        st.error("Risco alto — priorizar intervenção preventiva (engajamento/academia).")
    elif proba >= 0.40:
        st.warning("Risco moderado — monitorar e reforçar engajamento/rotina.")
    else:
        st.success("Risco baixo — manter acompanhamento e plano atual.")

    thr = meta.get("defas_threshold_p75")
    if thr is not None:
        st.caption(f"Alvo do treino: Defas >= P75 (threshold={thr}).")
        