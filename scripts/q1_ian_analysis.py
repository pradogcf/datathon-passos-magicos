"""
Pergunta 1 — Adequação do nível (IAN)
Gera:
- Gráficos em reports/q1/*.png
- Tabelas em reports/q1/*.csv
- Resumo em reports/q1/README_q1.md

Como rodar (na raiz do repo):
    python3 scripts/q1_ian_analysis.py
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =========================
# Config
# =========================
REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = REPO_ROOT / "data" / "raw" / "BASE_DE_DADOS_PEDE_2024_DATATHON.xlsx"
OUT_DIR = REPO_ROOT / "reports" / "q1"

OUT_DIR.mkdir(parents=True, exist_ok=True)


def _safe_to_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def main() -> None:
    # =========================================
    # 1) Carregar dados
    # =========================================
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Não encontrei o dataset em: {DATA_PATH}\n"
            f"Confirme se o arquivo está em data/raw/ com esse nome."
        )

    df = pd.read_excel(DATA_PATH)

    # Tipos numéricos relevantes para Q1
    df = _safe_to_numeric(df, ["IAN", "Defas", "Ano ingresso", "Idade 22", "Fase", "INDE 22"])

    # Checagens mínimas
    required_cols = ["IAN", "Defas", "Ano ingresso", "Idade 22", "Fase", "Pedra 22"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Colunas esperadas ausentes: {missing}. Colunas disponíveis: {list(df.columns)}")

    # Overview simples (para auditoria)
    overview = pd.DataFrame(
        {
            "coluna": df.columns,
            "dtype": [str(df[c].dtype) for c in df.columns],
            "missing_%": [round(df[c].isna().mean() * 100, 2) for c in df.columns],
            "n_unicos": [df[c].nunique(dropna=True) for c in df.columns],
        }
    ).sort_values(["missing_%", "coluna"], ascending=[False, True])
    overview.to_csv(OUT_DIR / "overview.csv", index=False)

    # =========================================
    # 2) Distribuição do IAN
    # =========================================
    ian = df["IAN"].dropna()

    # Histograma
    plt.figure(figsize=(10, 5))
    plt.hist(ian, bins=12)
    plt.title("Distribuição do IAN (Adequação ao Nível)")
    plt.xlabel("IAN")
    plt.ylabel("Quantidade de alunos")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "01_hist_ian.png", dpi=150)
    plt.close()

    # Boxplot
    plt.figure(figsize=(8, 4))
    plt.boxplot(ian, vert=False)
    plt.title("Boxplot do IAN")
    plt.xlabel("IAN")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "02_boxplot_ian.png", dpi=150)
    plt.close()

    # Frequência por valor
    ian_counts = ian.value_counts().sort_index()
    ian_counts_pct = (ian_counts / len(df) * 100).round(1)

    freq_ian = pd.DataFrame(
        {"IAN": ian_counts.index, "qtd": ian_counts.values, "%_total": ian_counts_pct.values}
    )
    freq_ian.to_csv(OUT_DIR / "freq_ian.csv", index=False)

    # =========================================
    # 3) Cálculos descritivos
    # =========================================
    desc = ian.describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9]).to_frame("IAN")
    desc.loc["missing"] = df["IAN"].isna().sum()
    desc.to_csv(OUT_DIR / "descritivos_ian.csv")

    # =========================================
    # 4) Evolução no tempo (proxy)
    # Observação: este dataset tem IAN em coluna única,
    # então usamos proxies temporais (coorte/idade).
    # =========================================
    ian_by_cohort = (
        df.groupby("Ano ingresso")["IAN"]
        .agg(["count", "mean", "median"])
        .reset_index()
        .sort_values("Ano ingresso")
    )
    ian_by_cohort.to_csv(OUT_DIR / "ian_por_coorte.csv", index=False)

    plt.figure(figsize=(10, 5))
    plt.plot(ian_by_cohort["Ano ingresso"], ian_by_cohort["mean"], marker="o")
    plt.title("IAN médio por coorte (Ano de ingresso) — proxy de evolução")
    plt.xlabel("Ano de ingresso na Passos Mágicos")
    plt.ylabel("IAN médio")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "03_ian_por_coorte.png", dpi=150)
    plt.close()

    ian_by_age = (
        df.groupby("Idade 22")["IAN"]
        .agg(["count", "mean", "median"])
        .reset_index()
        .sort_values("Idade 22")
    )
    ian_by_age.to_csv(OUT_DIR / "ian_por_idade.csv", index=False)

    plt.figure(figsize=(10, 5))
    plt.plot(ian_by_age["Idade 22"], ian_by_age["mean"], marker="o")
    plt.title("IAN médio por idade (2022) — visão complementar")
    plt.xlabel("Idade em 2022")
    plt.ylabel("IAN médio")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "04_ian_por_idade.png", dpi=150)
    plt.close()

    # =========================================
    # 5) Taxa de defasagem (aplicável via Defas)
    # Interpretação operacional (defensável):
    #   Defas = 0 -> sem defasagem
    #   Defas = -1 -> leve
    #   Defas <= -2 -> moderada+
    #   Defas <= -3 -> severa
    # =========================================
    defas = df["Defas"].dropna()

    rates = pd.DataFrame(
        {
            "grupo": [
                "Sem defasagem (Defas=0)",
                "Defasagem leve (Defas=-1)",
                "Defasagem moderada+ (Defas<=-2)",
                "Defasagem severa (Defas<=-3)",
            ],
            "qtd": [
                int((defas == 0).sum()),
                int((defas == -1).sum()),
                int((defas <= -2).sum()),
                int((defas <= -3).sum()),
            ],
        }
    )
    rates["%"] = (rates["qtd"] / len(defas) * 100).round(1)
    rates.to_csv(OUT_DIR / "taxas_defasagem.csv", index=False)

    plt.figure(figsize=(10, 5))
    plt.bar(rates["grupo"], rates["%"])
    plt.title("Taxa de defasagem (proxy via coluna Defas)")
    plt.ylabel("% de alunos")
    plt.xticks(rotation=20, ha="right")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "05_taxa_defasagem.png", dpi=150)
    plt.close()

    defas_counts = defas.value_counts().sort_index()
    pd.DataFrame({"Defas": defas_counts.index, "qtd": defas_counts.values}).to_csv(
        OUT_DIR / "distribuicao_defas.csv", index=False
    )

    plt.figure(figsize=(8, 4))
    plt.bar(defas_counts.index.astype(str), defas_counts.values)
    plt.title("Distribuição do Defas (diferença de nível)")
    plt.xlabel("Defas")
    plt.ylabel("Quantidade de alunos")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "06_distribuicao_defas.png", dpi=150)
    plt.close()

    # =========================================
    # 6) Análise por fase (Fase numérica e Pedra 22)
    # =========================================
    ian_by_fase = (
        df.groupby("Fase")["IAN"]
        .agg(["count", "mean", "median"])
        .reset_index()
        .sort_values("Fase")
    )
    ian_by_fase.to_csv(OUT_DIR / "ian_por_fase.csv", index=False)

    plt.figure(figsize=(10, 5))
    plt.bar(ian_by_fase["Fase"].astype(str), ian_by_fase["mean"])
    plt.title("IAN médio por Fase")
    plt.xlabel("Fase")
    plt.ylabel("IAN médio")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "07_ian_por_fase.png", dpi=150)
    plt.close()

    ian_by_pedra = (
        df.groupby("Pedra 22")["IAN"]
        .agg(["count", "mean", "median"])
        .reset_index()
        .sort_values("mean")
    )
    ian_by_pedra.to_csv(OUT_DIR / "ian_por_pedra.csv", index=False)

    plt.figure(figsize=(9, 5))
    plt.bar(ian_by_pedra["Pedra 22"], ian_by_pedra["mean"])
    plt.title("IAN médio por Pedra (classificação INDE)")
    plt.xlabel("Pedra 22")
    plt.ylabel("IAN médio")
    plt.xticks(rotation=15)
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "08_ian_por_pedra.png", dpi=150)
    plt.close()

    # Taxa de defasagem moderada+ por fase
    tmp = df.copy()
    tmp["defas_moderada_mais"] = (tmp["Defas"] <= -2).astype(int)
    rate_defas_by_fase = tmp.groupby("Fase")["defas_moderada_mais"].mean().reset_index()
    rate_defas_by_fase["%"] = (rate_defas_by_fase["defas_moderada_mais"] * 100).round(1)
    rate_defas_by_fase.to_csv(OUT_DIR / "taxa_defas_moderada_por_fase.csv", index=False)

    plt.figure(figsize=(10, 5))
    plt.bar(rate_defas_by_fase["Fase"].astype(str), rate_defas_by_fase["%"])
    plt.title("Taxa de defasagem moderada+ (Defas<=-2) por Fase")
    plt.xlabel("Fase")
    plt.ylabel("% de alunos")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "09_defas_moderada_por_fase.png", dpi=150)
    plt.close()

    # =========================================
    # Resumo final (markdown)
    # =========================================
    # Números-chave:
    pct_ian = dict(zip(freq_ian["IAN"], freq_ian["%_total"]))
    pct_moderada = float(rates.loc[rates["grupo"].str.contains("moderada", case=False), "%"].iloc[0])
    pct_severa = float(rates.loc[rates["grupo"].str.contains("severa", case=False), "%"].iloc[0])

    md = f"""# Pergunta 1 — Adequação do nível (IAN)

## Estrutura aplicada
1) Carregar dados  
2) Distribuição do IAN  
3) Cálculos descritivos  
4) Evolução no tempo (proxy)  
5) Taxa de defasagem (se aplicável)  
6) Análise por fase  

## Observação metodológica (importante)
O dataset disponibilizado possui **uma coluna única `IAN`** (sem sufixo de ano e sem granularidade intra-ano).  
Por isso, a análise de “evolução no tempo” foi feita por **proxies**:
- **Ano de ingresso** (coorte/tempo no programa)
- **Idade em 2022** (visão complementar)

## Resultados principais
- Distribuição do IAN:
  - IAN = 2,5 → **{pct_ian.get(2.5, np.nan)}%**
  - IAN = 5,0 → **{pct_ian.get(5.0, np.nan)}%**
  - IAN = 10,0 → **{pct_ian.get(10.0, np.nan)}%**

- Taxa de defasagem (proxy via `Defas`):
  - Defasagem **moderada+ (Defas <= -2)**: **{pct_moderada}%**
  - Defasagem **severa (Defas <= -3)**: **{pct_severa}%**

- Análise por fase e progressão:
  - O IAN médio aumenta conforme a **Pedra 22** evolui (Quartzo → Topázio), reforçando consistência com a progressão do programa.

## Conclusão (resposta da Pergunta 1)
O perfil de adequação ao nível mostra concentração em IAN=5, com um grupo relevante em IAN=10 e uma minoria em IAN=2,5.  
Pelo indicador de defasagem (`Defas`), aproximadamente **{pct_moderada}%** dos alunos estão em defasagem **moderada ou pior** e **{pct_severa}%** em defasagem **severa**.  
Além disso, a adequação cresce de forma consistente conforme a classificação geral do programa (Pedra 22), com melhores resultados nas pedras mais avançadas.
"""

    (OUT_DIR / "README_q1.md").write_text(md, encoding="utf-8")

    print("✅ Q1 concluída! Arquivos gerados em:", OUT_DIR)
    print("   - Gráficos: reports/q1/*.png")
    print("   - Tabelas:  reports/q1/*.csv")
    print("   - Resumo:   reports/q1/README_q1.md")


if __name__ == "__main__":
    main()
