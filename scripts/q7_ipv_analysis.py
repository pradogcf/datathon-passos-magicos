"""
Pergunta 7 — Ponto de Virada (IPV)

Objetivo:
Identificar quais indicadores mais influenciam o IPV
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/raw/BASE_DE_DADOS_PEDE_2024_DATATHON.xlsx"
OUT = ROOT / "reports/q7"
OUT.mkdir(parents=True, exist_ok=True)

def main():

    df = pd.read_excel(DATA)

    cols = ["IPV","IDA","IEG","IAA","IPS","IAN","Fase","Pedra 22"]
    df = df[cols]

    for c in cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # =====================================================
    # 1 — Distribuição IPV
    # =====================================================
    ipv = df["IPV"].dropna()

    plt.hist(ipv, bins=20)
    plt.title("Distribuição do IPV")
    plt.savefig(OUT/"01_hist_ipv.png")
    plt.close()

    desc = ipv.describe()
    desc.to_csv(OUT/"descritivos_ipv.csv")

    p25 = ipv.quantile(0.25)
    p75 = ipv.quantile(0.75)

    # =====================================================
    # 2 — Evolução proxy por fase
    # =====================================================
    fase = df.groupby("Fase")["IPV"].mean().reset_index()
    fase.to_csv(OUT/"ipv_por_fase.csv")

    plt.bar(fase["Fase"], fase["IPV"])
    plt.title("IPV médio por fase")
    plt.savefig(OUT/"02_ipv_por_fase.png")
    plt.close()

    # =====================================================
    # 3 — Correlações
    # =====================================================
    indicadores = ["IDA","IEG","IAA","IPS","IAN"]

    corr = []

    for ind in indicadores:
        corr.append([
            ind,
            df[ind].corr(df["IPV"]),
            df[ind].corr(df["IPV"], method="spearman")
        ])

    corr = pd.DataFrame(corr, columns=["Indicador","Pearson","Spearman"])
    corr = corr.sort_values("Pearson", ascending=False)
    corr.to_csv(OUT/"correlacoes_ipv.csv", index=False)

    # =====================================================
    # 4 — Ranking gráfico
    # =====================================================
    plt.bar(corr["Indicador"], corr["Pearson"])
    plt.title("Importância dos fatores para IPV")
    plt.savefig(OUT/"03_importancia_ipv.png")
    plt.close()

    # =====================================================
    # 5 — Grupos alto vs baixo IPV
    # =====================================================
    df["alto_ipv"] = df["IPV"] >= p75
    df["baixo_ipv"] = df["IPV"] <= p25

    comp = pd.DataFrame({
        "Indicador": indicadores,
        "Diferença Alto-Baixo IPV": [
            df[df["alto_ipv"]][i].mean() - df[df["baixo_ipv"]][i].mean()
            for i in indicadores
        ]
    })

    comp.to_csv(OUT/"comparacao_grupos_ipv.csv", index=False)

    # =====================================================
    # 6 — README automático
    # =====================================================
    md = f"""
# Pergunta 7 — Ponto de Virada (IPV)

## Percentis
P25 = {round(p25,2)}
P75 = {round(p75,2)}

## Ranking de influência (Pearson)
{corr.to_string(index=False)}

## Diferença Alto vs Baixo IPV
{comp.to_string(index=False)}

## Conclusão
Os indicadores com maior correlação positiva e maior diferença entre grupos
são os que mais influenciam o ponto de virada dos alunos.
"""

    (OUT/"README_q7.md").write_text(md)

    print("Q7 concluída!")

if __name__ == "__main__":
    main()