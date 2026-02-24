"""
Pergunta 5 — Aspectos Psicossociais (IPS)

Objetivo:
Identificar se padrões psicossociais antecedem queda de desempenho ou engajamento
"""

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/raw/BASE_DE_DADOS_PEDE_2024_DATATHON.xlsx"
OUT = ROOT / "reports/q5"
OUT.mkdir(parents=True, exist_ok=True)

def main():

    df = pd.read_excel(DATA)

    for c in ["IPS","IDA","IEG","IAN","Fase","Pedra 22"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # =====================================================
    # 1 — Distribuição
    # =====================================================
    ips = df["IPS"].dropna()

    plt.hist(ips, bins=20)
    plt.title("Distribuição do IPS")
    plt.savefig(OUT/"01_hist_ips.png")
    plt.close()

    # =====================================================
    # 2 — Estatísticas
    # =====================================================
    desc = ips.describe()
    desc.to_csv(OUT/"descritivos_ips.csv")

    p25 = ips.quantile(0.25)
    p75 = ips.quantile(0.75)

    # =====================================================
    # 3 — Evolução proxy
    # =====================================================
    fase = df.groupby("Fase")["IPS"].mean().reset_index()
    fase.to_csv(OUT/"ips_por_fase.csv")

    plt.bar(fase["Fase"], fase["IPS"])
    plt.title("IPS médio por fase")
    plt.savefig(OUT/"02_ips_por_fase.png")
    plt.close()

    # =====================================================
    # 4 — Correlações
    # =====================================================
    corr = pd.DataFrame({
        "Variavel":["IPS x IDA","IPS x IEG","IPS x IAN"],
        "Pearson":[
            df["IPS"].corr(df["IDA"]),
            df["IPS"].corr(df["IEG"]),
            df["IPS"].corr(df["IAN"])
        ]
    })

    corr.to_csv(OUT/"correlacoes.csv",index=False)

    # =====================================================
    # 5 — Análise de risco
    # =====================================================
    df["baixo_ips"] = df["IPS"] <= p25
    df["alto_ips"] = df["IPS"] >= p75

    comp = pd.DataFrame({
        "Grupo":["Baixo IPS","Alto IPS"],
        "IDA":[
            df[df["baixo_ips"]]["IDA"].mean(),
            df[df["alto_ips"]]["IDA"].mean()
        ],
        "IEG":[
            df[df["baixo_ips"]]["IEG"].mean(),
            df[df["alto_ips"]]["IEG"].mean()
        ],
        "IAN":[
            df[df["baixo_ips"]]["IAN"].mean(),
            df[df["alto_ips"]]["IAN"].mean()
        ]
    })

    comp.to_csv(OUT/"comparacao_grupos.csv",index=False)

    # =====================================================
    # 6 — README automático
    # =====================================================
    md = f"""
# Pergunta 5 — Aspectos Psicossociais

## Percentis
P25 = {round(p25,2)}
P75 = {round(p75,2)}

## Correlações
IPS x IDA = {round(df["IPS"].corr(df["IDA"]),3)}
IPS x IEG = {round(df["IPS"].corr(df["IEG"]),3)}
IPS x IAN = {round(df["IPS"].corr(df["IAN"]),3)}

## Diferença entre grupos
IDA alto-baixo = {round(comp.iloc[1,1]-comp.iloc[0,1],2)}
IEG alto-baixo = {round(comp.iloc[1,2]-comp.iloc[0,2],2)}
IAN alto-baixo = {round(comp.iloc[1,3]-comp.iloc[0,3],2)}
"""

    (OUT/"README_q5.md").write_text(md)

    print("Q5 concluída!")

if __name__ == "__main__":
    main()