"""
Pergunta 4 — Autoavaliação (IAA)
Objetivo: verificar se percepção do aluno é coerente com desempenho real.

Como rodar:
python scripts/q4_iaa_analysis.py
"""

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/raw/BASE_DE_DADOS_PEDE_2024_DATATHON.xlsx"
OUT = ROOT / "reports/q4"
OUT.mkdir(parents=True, exist_ok=True)


def main():

    df = pd.read_excel(DATA)

    for c in ["IAA","IDA","IEG","Fase","Ano ingresso","Pedra 22"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # ======================================================
    # 1 — Distribuição
    # ======================================================
    iaa = df["IAA"].dropna()

    plt.hist(iaa, bins=20)
    plt.title("Distribuição do IAA")
    plt.savefig(OUT/"01_hist_iaa.png")
    plt.close()

    # ======================================================
    # 2 — Descritivos
    # ======================================================
    desc = iaa.describe()
    desc.to_csv(OUT/"descritivos_iaa.csv")

    p25 = iaa.quantile(0.25)
    p75 = iaa.quantile(0.75)

    # ======================================================
    # 3 — Evolução proxy
    # ======================================================
    fase = df.groupby("Fase")["IAA"].mean().reset_index()
    fase.to_csv(OUT/"iaa_por_fase.csv")

    plt.bar(fase["Fase"], fase["IAA"])
    plt.title("IAA por Fase")
    plt.savefig(OUT/"02_iaa_por_fase.png")
    plt.close()

    # ======================================================
    # 4 — Correlações (CORE)
    # ======================================================
    corr = pd.DataFrame({
        "Variavel":["IAA x IDA","IAA x IEG"],
        "Pearson":[
            df["IAA"].corr(df["IDA"]),
            df["IAA"].corr(df["IEG"])
        ],
        "Spearman":[
            df["IAA"].corr(df["IDA"],method="spearman"),
            df["IAA"].corr(df["IEG"],method="spearman")
        ]
    })

    corr.to_csv(OUT/"correlacoes.csv",index=False)

    # ======================================================
    # 5 — Coerência percepção vs realidade
    # ======================================================
    df["baixo_iaa"] = (df["IAA"]<=p25)
    df["alto_iaa"] = (df["IAA"]>=p75)

    comp = pd.DataFrame({
        "Grupo":["Baixo IAA","Alto IAA"],
        "IDA_medio":[
            df[df["baixo_iaa"]]["IDA"].mean(),
            df[df["alto_iaa"]]["IDA"].mean()
        ],
        "IEG_medio":[
            df[df["baixo_iaa"]]["IEG"].mean(),
            df[df["alto_iaa"]]["IEG"].mean()
        ]
    })

    comp.to_csv(OUT/"comparacao_grupos.csv",index=False)

    # ======================================================
    # 6 — README automático
    # ======================================================
    md = f"""
# Pergunta 4 — Autoavaliação

## Percentis
P25 = {round(p25,2)}
P75 = {round(p75,2)}

## Correlações
IAA x IDA = {round(df["IAA"].corr(df["IDA"]),3)}
IAA x IEG = {round(df["IAA"].corr(df["IEG"]),3)}

## Diferença entre grupos
IDA médio (alto - baixo) = {round(comp.iloc[1,1]-comp.iloc[0,1],2)}
IEG médio (alto - baixo) = {round(comp.iloc[1,2]-comp.iloc[0,2],2)}
"""

    (OUT/"README_q4.md").write_text(md)

    print("Q4 concluída!")


if __name__ == "__main__":
    main()