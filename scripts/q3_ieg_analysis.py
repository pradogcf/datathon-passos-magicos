"""
Pergunta 3 — Engajamento nas atividades (IEG)
Objetivo: avaliar distribuição do IEG e sua relação com IDA e IPV.

Gera:
- Gráficos em reports/q3/*.png
- Tabelas em reports/q3/*.csv
- Resumo em reports/q3/README_q3.md

Como rodar (na raiz do repo):
    python3 scripts/q3_ieg_analysis.py
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = REPO_ROOT / "data" / "raw" / "BASE_DE_DADOS_PEDE_2024_DATATHON.xlsx"
OUT_DIR = REPO_ROOT / "reports" / "q3"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _safe_to_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def _corr_safe(a: pd.Series, b: pd.Series, method: str) -> float:
    x = pd.concat([a, b], axis=1).dropna()
    if x.shape[0] < 5:
        return float("nan")
    return float(x.corr(method=method).iloc[0, 1])


def main() -> None:
    # =========================================================
    # 1) Carregar dados
    # =========================================================
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Não encontrei o dataset em: {DATA_PATH}\n"
            f"Confirme se o arquivo está em data/raw/ com esse nome."
        )

    df = pd.read_excel(DATA_PATH)

    # Converte indicadores comuns (se existirem)
    df = _safe_to_numeric(df, ["IEG", "IDA", "IPV", "IAN", "IAA", "IPS", "IPP", "Defas",
                               "Ano ingresso", "Idade 22", "Fase", "INDE 22"])

    required = ["IEG", "IDA", "IPV", "Ano ingresso", "Idade 22", "Fase", "Pedra 22"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Colunas esperadas ausentes: {missing}. Colunas disponíveis: {list(df.columns)}")

    # Overview
    overview = pd.DataFrame(
        {
            "coluna": df.columns,
            "dtype": [str(df[c].dtype) for c in df.columns],
            "missing_%": [round(df[c].isna().mean() * 100, 2) for c in df.columns],
            "n_unicos": [df[c].nunique(dropna=True) for c in df.columns],
        }
    ).sort_values(["missing_%", "coluna"], ascending=[False, True])
    overview.to_csv(OUT_DIR / "overview.csv", index=False)

    # =========================================================
    # 2) Distribuição do IEG
    # =========================================================
    ieg = df["IEG"].dropna()

    plt.figure(figsize=(10, 5))
    plt.hist(ieg, bins=20)
    plt.title("Distribuição do IEG (Engajamento)")
    plt.xlabel("IEG")
    plt.ylabel("Quantidade de alunos")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "01_hist_ieg.png", dpi=150)
    plt.close()

    plt.figure(figsize=(8, 4))
    plt.boxplot(ieg, vert=False)
    plt.title("Boxplot do IEG")
    plt.xlabel("IEG")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "02_boxplot_ieg.png", dpi=150)
    plt.close()

    # Frequência por faixas (percentis)
    bins = np.quantile(ieg, [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0])
    bins = np.unique(np.round(bins, 6))
    if len(bins) < 4:
        bins = np.linspace(float(ieg.min()), float(ieg.max()), 6)

    ieg_bins = pd.cut(ieg, bins=bins, include_lowest=True)
    freq_bins = ieg_bins.value_counts().sort_index()
    freq_bins_pct = (freq_bins / df["IEG"].notna().sum() * 100).round(1)

    freq_ieg_bins = pd.DataFrame(
        {"faixa_ieg": freq_bins.index.astype(str), "qtd": freq_bins.values, "%_nao_nulo": freq_bins_pct.values}
    )
    freq_ieg_bins.to_csv(OUT_DIR / "freq_ieg_por_faixa.csv", index=False)

    # =========================================================
    # 3) Cálculos descritivos
    # =========================================================
    desc = ieg.describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9]).to_frame("IEG")
    desc.loc["missing"] = df["IEG"].isna().sum()
    desc.to_csv(OUT_DIR / "descritivos_ieg.csv")

    p10 = float(ieg.quantile(0.10))
    p25 = float(ieg.quantile(0.25))
    p50 = float(ieg.quantile(0.50))
    p75 = float(ieg.quantile(0.75))

    # =========================================================
    # 4) Evolução no tempo (proxy)
    # Sem granularidade intra-ano → usar coorte/idade/fase/pedra
    # =========================================================
    ieg_by_cohort = (
        df.groupby("Ano ingresso")["IEG"]
        .agg(["count", "mean", "median"])
        .reset_index()
        .sort_values("Ano ingresso")
    )
    ieg_by_cohort.to_csv(OUT_DIR / "ieg_por_coorte.csv", index=False)

    plt.figure(figsize=(10, 5))
    plt.plot(ieg_by_cohort["Ano ingresso"], ieg_by_cohort["mean"], marker="o")
    plt.title("IEG médio por coorte (Ano de ingresso) — proxy de evolução")
    plt.xlabel("Ano de ingresso na Passos Mágicos")
    plt.ylabel("IEG médio")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "03_ieg_por_coorte.png", dpi=150)
    plt.close()

    ieg_by_age = (
        df.groupby("Idade 22")["IEG"]
        .agg(["count", "mean", "median"])
        .reset_index()
        .sort_values("Idade 22")
    )
    ieg_by_age.to_csv(OUT_DIR / "ieg_por_idade.csv", index=False)

    plt.figure(figsize=(10, 5))
    plt.plot(ieg_by_age["Idade 22"], ieg_by_age["mean"], marker="o")
    plt.title("IEG médio por idade (2022) — visão complementar")
    plt.xlabel("Idade em 2022")
    plt.ylabel("IEG médio")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "04_ieg_por_idade.png", dpi=150)
    plt.close()

    # =========================================================
    # 5) Taxas + Relação com IDA e IPV (núcleo da Pergunta 3)
    # Critérios:
    # - Baixo engajamento: IEG <= P25
    # - Alto engajamento: IEG >= P75
    # E medir impacto em IDA e IPV
    # =========================================================
    tmp = df.copy()
    tmp["baixo_ieg_p25"] = (tmp["IEG"] <= p25).astype(int)
    tmp["alto_ieg_p75"] = (tmp["IEG"] >= p75).astype(int)

    rate_ieg = pd.DataFrame(
        {
            "criterio": ["Baixo engajamento (IEG <= P25)", "Alto engajamento (IEG >= P75)"],
            "corte": [round(p25, 6), round(p75, 6)],
            "qtd": [int(tmp["baixo_ieg_p25"].sum()), int(tmp["alto_ieg_p75"].sum())],
        }
    )
    base_n = int(tmp["IEG"].notna().sum())
    rate_ieg["%"] = (rate_ieg["qtd"] / base_n * 100).round(1)
    rate_ieg.to_csv(OUT_DIR / "taxa_baixo_alto_engajamento.csv", index=False)

    plt.figure(figsize=(9, 5))
    plt.bar(rate_ieg["criterio"], rate_ieg["%"])
    plt.title("Taxa de baixo vs alto engajamento (cortes por percentil)")
    plt.ylabel("% dos alunos com IEG não-nulo")
    plt.xticks(rotation=10, ha="right")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "05_taxa_baixo_alto_engajamento.png", dpi=150)
    plt.close()

    # Correlações (Pearson e Spearman) IEG vs IDA/IPV
    corr_ieg_ida_p = _corr_safe(tmp["IEG"], tmp["IDA"], "pearson")
    corr_ieg_ida_s = _corr_safe(tmp["IEG"], tmp["IDA"], "spearman")
    corr_ieg_ipv_p = _corr_safe(tmp["IEG"], tmp["IPV"], "pearson")
    corr_ieg_ipv_s = _corr_safe(tmp["IEG"], tmp["IPV"], "spearman")

    corr_tbl = pd.DataFrame(
        {
            "par": ["IEG x IDA", "IEG x IPV"],
            "pearson": [corr_ieg_ida_p, corr_ieg_ipv_p],
            "spearman": [corr_ieg_ida_s, corr_ieg_ipv_s],
        }
    )
    corr_tbl.to_csv(OUT_DIR / "correlacoes_ieg_ida_ipv.csv", index=False)

    # Scatter plots (se dados suficientes)
    scat = tmp[["IEG", "IDA", "IPV"]].dropna()
    if scat.shape[0] >= 10:
        plt.figure(figsize=(7, 5))
        plt.scatter(scat["IEG"], scat["IDA"])
        plt.title("Dispersão: IEG vs IDA")
        plt.xlabel("IEG (Engajamento)")
        plt.ylabel("IDA (Desempenho Acadêmico)")
        plt.grid(True, alpha=0.2)
        plt.tight_layout()
        plt.savefig(OUT_DIR / "06_scatter_ieg_vs_ida.png", dpi=150)
        plt.close()

        plt.figure(figsize=(7, 5))
        plt.scatter(scat["IEG"], scat["IPV"])
        plt.title("Dispersão: IEG vs IPV")
        plt.xlabel("IEG (Engajamento)")
        plt.ylabel("IPV (Ponto de Virada)")
        plt.grid(True, alpha=0.2)
        plt.tight_layout()
        plt.savefig(OUT_DIR / "07_scatter_ieg_vs_ipv.png", dpi=150)
        plt.close()

    # Comparação de médias (alto vs baixo engajamento)
    low = tmp[tmp["baixo_ieg_p25"] == 1]
    high = tmp[tmp["alto_ieg_p75"] == 1]

    comp = pd.DataFrame(
        {
            "grupo": ["Baixo engajamento (<=P25)", "Alto engajamento (>=P75)"],
            "n": [int(low["IEG"].notna().sum()), int(high["IEG"].notna().sum())],
            "IDA_medio": [float(low["IDA"].mean()), float(high["IDA"].mean())],
            "IPV_medio": [float(low["IPV"].mean()), float(high["IPV"].mean())],
        }
    )
    comp["delta_IDA"] = comp.loc[1, "IDA_medio"] - comp.loc[0, "IDA_medio"]
    comp["delta_IPV"] = comp.loc[1, "IPV_medio"] - comp.loc[0, "IPV_medio"]
    comp.to_csv(OUT_DIR / "comparacao_alto_vs_baixo_engajamento.csv", index=False)

    # Boxplots comparativos (IDA e IPV por grupo de engajamento)
    grp = tmp.copy()
    grp["grupo_engajamento"] = np.where(grp["IEG"] <= p25, "Baixo (<=P25)",
                               np.where(grp["IEG"] >= p75, "Alto (>=P75)", "Intermediario"))
    grp2 = grp[grp["grupo_engajamento"].isin(["Baixo (<=P25)", "Alto (>=P75)"])]

    # IDA boxplot
    data_ida = [grp2.loc[grp2["grupo_engajamento"] == "Baixo (<=P25)", "IDA"].dropna(),
                grp2.loc[grp2["grupo_engajamento"] == "Alto (>=P75)", "IDA"].dropna()]
    plt.figure(figsize=(8, 4))
    plt.boxplot(data_ida, labels=["Baixo IEG", "Alto IEG"])
    plt.title("IDA por grupo de engajamento (Baixo vs Alto IEG)")
    plt.ylabel("IDA")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "08_box_ida_baixo_vs_alto_ieg.png", dpi=150)
    plt.close()

    # IPV boxplot
    data_ipv = [grp2.loc[grp2["grupo_engajamento"] == "Baixo (<=P25)", "IPV"].dropna(),
                grp2.loc[grp2["grupo_engajamento"] == "Alto (>=P75)", "IPV"].dropna()]
    plt.figure(figsize=(8, 4))
    plt.boxplot(data_ipv, labels=["Baixo IEG", "Alto IEG"])
    plt.title("IPV por grupo de engajamento (Baixo vs Alto IEG)")
    plt.ylabel("IPV")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "09_box_ipv_baixo_vs_alto_ieg.png", dpi=150)
    plt.close()

    # =========================================================
    # 6) Análise por fase / pedra
    # =========================================================
    ieg_by_fase = (
        df.groupby("Fase")["IEG"]
        .agg(["count", "mean", "median"])
        .reset_index()
        .sort_values("Fase")
    )
    ieg_by_fase.to_csv(OUT_DIR / "ieg_por_fase.csv", index=False)

    plt.figure(figsize=(10, 5))
    plt.bar(ieg_by_fase["Fase"].astype(str), ieg_by_fase["mean"])
    plt.title("IEG médio por Fase")
    plt.xlabel("Fase")
    plt.ylabel("IEG médio")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "10_ieg_por_fase.png", dpi=150)
    plt.close()

    ieg_by_pedra = (
        df.groupby("Pedra 22")["IEG"]
        .agg(["count", "mean", "median"])
        .reset_index()
        .sort_values("mean")
    )
    ieg_by_pedra.to_csv(OUT_DIR / "ieg_por_pedra.csv", index=False)

    plt.figure(figsize=(9, 5))
    plt.bar(ieg_by_pedra["Pedra 22"], ieg_by_pedra["mean"])
    plt.title("IEG médio por Pedra (classificação INDE)")
    plt.xlabel("Pedra 22")
    plt.ylabel("IEG médio")
    plt.xticks(rotation=15)
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "11_ieg_por_pedra.png", dpi=150)
    plt.close()

    # Taxa de baixo engajamento por fase
    rate_low_by_fase = (
        tmp.groupby("Fase")["baixo_ieg_p25"].mean().reset_index().rename(columns={"baixo_ieg_p25": "taxa"})
    )
    rate_low_by_fase["%"] = (rate_low_by_fase["taxa"] * 100).round(1)
    rate_low_by_fase.to_csv(OUT_DIR / "taxa_baixo_engajamento_por_fase.csv", index=False)

    plt.figure(figsize=(10, 5))
    plt.bar(rate_low_by_fase["Fase"].astype(str), rate_low_by_fase["%"])
    plt.title("Taxa de baixo engajamento (IEG <= P25) por Fase")
    plt.xlabel("Fase")
    plt.ylabel("% de alunos")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "12_taxa_baixo_engajamento_por_fase.png", dpi=150)
    plt.close()

    # =========================================================
    # Conclusão automática (README)
    # =========================================================
    md = f"""# Pergunta 3 — Engajamento nas atividades (IEG)

## Estrutura aplicada
1) Carregar dados  
2) Distribuição do IEG  
3) Cálculos descritivos  
4) Evolução no tempo (proxy)  
5) Taxas + relação com IDA/IPV  
6) Análise por fase/pedra  

## Observação metodológica (importante)
O dataset disponibilizado possui **uma coluna única `IEG`** (sem sufixo de ano e sem granularidade intra-ano).
A “evolução no tempo” é avaliada via **proxies** (coorte, idade, fase e pedra).

## Descritivos e cortes (percentis)
- P10 = **{round(p10,6)}**
- P25 = **{round(p25,6)}**
- Mediana (P50) = **{round(p50,6)}**
- P75 = **{round(p75,6)}**

## Taxas (engajamento)
- Baixo engajamento (IEG <= P25): **{rate_ieg.loc[0, '%']}%**
- Alto engajamento (IEG >= P75): **{rate_ieg.loc[1, '%']}%**

## Relação IEG ↔ IDA / IPV (core da Pergunta 3)
**Correlações (tabela completa em `correlacoes_ieg_ida_ipv.csv`):**
- IEG x IDA → Pearson: **{corr_ieg_ida_p:.4f}** | Spearman: **{corr_ieg_ida_s:.4f}**
- IEG x IPV → Pearson: **{corr_ieg_ipv_p:.4f}** | Spearman: **{corr_ieg_ipv_s:.4f}**

**Comparação Alto vs Baixo engajamento (tabela em `comparacao_alto_vs_baixo_engajamento.csv`):**
- IDA médio (Alto IEG) - (Baixo IEG) = **{comp.loc[0,'delta_IDA']:.4f}**
- IPV médio (Alto IEG) - (Baixo IEG) = **{comp.loc[0,'delta_IPV']:.4f}**

## Conclusão (resposta da Pergunta 3)
Os resultados mostram a distribuição do engajamento e permitem segmentar alunos em baixo e alto engajamento (P25/P75).
A análise de correlação e a comparação entre grupos (alto vs baixo IEG) indicam se existe uma relação consistente entre engajamento e desempenho acadêmico (IDA) e com o ponto de virada (IPV).
Caso as correlações sejam positivas e o grupo de alto engajamento apresente IDA/IPV médios maiores, isso sustenta a hipótese de que o engajamento é um fator-chave para evolução do aluno — e deve ser priorizado em intervenções.
"""
    (OUT_DIR / "README_q3.md").write_text(md, encoding="utf-8")

    print("✅ Q3 concluída! Arquivos gerados em:", OUT_DIR)
    print("   - Gráficos: reports/q3/*.png")
    print("   - Tabelas:  reports/q3/*.csv")
    print("   - Resumo:   reports/q3/README_q3.md")


if __name__ == "__main__":
    main()
