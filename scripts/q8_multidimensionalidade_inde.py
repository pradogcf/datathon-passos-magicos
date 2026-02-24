"""
Pergunta 8 — Multidimensionalidade dos indicadores

Objetivo:
Identificar quais combinações de indicadores melhor explicam o INDE
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/raw/BASE_DE_DADOS_PEDE_2024_DATATHON.xlsx"
OUT = ROOT / "reports/q8"
OUT.mkdir(parents=True, exist_ok=True)

def main():

    df = pd.read_excel(DATA)

    cols = ["INDE 22","IDA","IEG","IAA","IPS","IAN","Fase"]
    df = df[cols]

    for c in cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # =====================================================
    # 1 — Distribuição INDE
    # =====================================================
    inde = df["INDE 22"].dropna()

    plt.hist(inde, bins=20)
    plt.title("Distribuição do INDE")
    plt.savefig(OUT/"01_hist_inde.png")
    plt.close()

    desc = inde.describe()
    desc.to_csv(OUT/"descritivos_inde.csv")

    # =====================================================
    # 2 — Evolução por fase
    # =====================================================
    fase = df.groupby("Fase")["INDE 22"].mean().reset_index()
    fase.to_csv(OUT/"inde_por_fase.csv")

    plt.bar(fase["Fase"], fase["INDE 22"])
    plt.title("INDE médio por fase")
    plt.savefig(OUT/"02_inde_por_fase.png")
    plt.close()

    # =====================================================
    # 3 — Correlações
    # =====================================================
    indicadores = ["IDA","IEG","IAA","IPS","IAN"]

    corr = []

    for ind in indicadores:
        corr.append([
            ind,
            df[ind].corr(df["INDE 22"]),
            df[ind].corr(df["INDE 22"], method="spearman")
        ])

    corr = pd.DataFrame(corr, columns=["Indicador","Pearson","Spearman"])
    corr = corr.sort_values("Pearson", ascending=False)
    corr.to_csv(OUT/"correlacoes_inde.csv", index=False)

    # =====================================================
    # 4 — Regressão linear (importância conjunta)
    # =====================================================
    model_df = df.dropna()

    X = model_df[indicadores]
    y = model_df["INDE 22"]

    model = LinearRegression()
    model.fit(X, y)

    importancia = pd.DataFrame({
        "Indicador": indicadores,
        "Coeficiente": model.coef_
    }).sort_values("Coeficiente", ascending=False)

    importancia.to_csv(OUT/"importancia_regressao.csv", index=False)

    plt.bar(importancia["Indicador"], importancia["Coeficiente"])
    plt.title("Importância conjunta dos indicadores para INDE")
    plt.savefig(OUT/"03_importancia_regressao.png")
    plt.close()

    # =====================================================
    # 5 — Grupos alto vs baixo INDE
    # =====================================================
    p25 = inde.quantile(0.25)
    p75 = inde.quantile(0.75)

    df["alto_inde"] = df["INDE 22"] >= p75
    df["baixo_inde"] = df["INDE 22"] <= p25

    comp = pd.DataFrame({
        "Indicador": indicadores,
        "Diferença Alto-Baixo INDE": [
            df[df["alto_inde"]][i].mean() - df[df["baixo_inde"]][i].mean()
            for i in indicadores
        ]
    })

    comp.to_csv(OUT/"comparacao_grupos_inde.csv", index=False)

    # =====================================================
    # 6 — README automático
    # =====================================================
    md = f"""
# Pergunta 8 — Multidimensionalidade do INDE

## Ranking de influência (correlação)
{corr.to_string(index=False)}

## Importância conjunta (regressão)
{importancia.to_string(index=False)}

## Diferença Alto vs Baixo INDE
{comp.to_string(index=False)}

## Conclusão
Os indicadores com maior correlação e coeficientes positivos na regressão
são os que melhor explicam o desempenho global do aluno.
"""

    (OUT/"README_q8.md").write_text(md)

    print("Q8 concluída!")

if __name__ == "__main__":
    main()