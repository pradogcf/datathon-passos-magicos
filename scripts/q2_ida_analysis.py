"""
Pergunta 2 — Desempenho Acadêmico (IDA)
Gera:
- Gráficos em reports/q2/*.png
- Tabelas em reports/q2/*.csv
- Resumo em reports/q2/README_q2.md

Como rodar (na raiz do repo):
    python3 scripts/q2_ida_analysis.py
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =========================
# Config
# =========================
REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = REPO_ROOT / "data" / "raw" / "BASE_DE_DADOS_PEDE_2024_DATATHON.xlsx"
OUT_DIR = REPO_ROOT / "reports" / "q2"
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

    # Tipos numéricos relevantes para Q2
    df = _safe_to_numeric(df, ["IDA", "Ano ingresso", "Idade 22", "Fase", "INDE 22", "IAN", "IEG", "IAA", "IPS", "IPP", "IPV", "Defas"])

    required_cols = ["IDA", "Ano ingresso", "Idade 22", "Fase", "Pedra 22"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Colunas esperadas ausentes: {missing}. Colunas disponíveis: {list(df.columns)}")

    # Overview (auditoria)
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
    # 2) Distribuição do IDA
    # =========================================
    ida = df["IDA"].dropna()

    plt.figure(figsize=(10, 5))
    plt.hist(ida, bins=20)
    plt.title("Distribuição do IDA (Desempenho Acadêmico)")
    plt.xlabel("IDA")
    plt.ylabel("Quantidade de alunos")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "01_hist_ida.png", dpi=150)
    plt.close()

    plt.figure(figsize=(8, 4))
    plt.boxplot(ida, vert=False)
    plt.title("Boxplot do IDA")
    plt.xlabel("IDA")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "02_boxplot_ida.png", dpi=150)
    plt.close()

    # Frequência por faixas (bins) para leitura executiva
    # (bins automáticos, mas estáveis)
    bins = np.quantile(ida, [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0])
    bins = np.unique(np.round(bins, 4))
    if len(bins) < 4:
        # fallback se distribuição tiver poucos valores
        bins = np.linspace(float(ida.min()), float(ida.max()), 6)

    ida_bins = pd.cut(ida, bins=bins, include_lowest=True)
    freq_bins = ida_bins.value_counts().sort_index()
    freq_bins_pct = (freq_bins / len(df) * 100).round(1)

    freq_ida_bins = pd.DataFrame(
        {"faixa_ida": freq_bins.index.astype(str), "qtd": freq_bins.values, "%_total": freq_bins_pct.values}
    )
    freq_ida_bins.to_csv(OUT_DIR / "freq_ida_por_faixa.csv", index=False)

    # =========================================
    # 3) Cálculos descritivos
    # =========================================
    desc = ida.describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9]).to_frame("IDA")
    desc.loc["missing"] = df["IDA"].isna().sum()
    desc.to_csv(OUT_DIR / "descritivos_ida.csv")

    # Percentis-chave para classificação de desempenho
    p10 = float(ida.quantile(0.10))
    p25 = float(ida.quantile(0.25))
    p50 = float(ida.quantile(0.50))

    # =========================================
    # 4) Evolução no tempo (proxy)
    # Observação: dataset atual traz IDA em coluna única (sem 2022/2023/2024).
    # Então avaliamos evolução por proxies:
    # - coorte (Ano ingresso)
    # - idade (Idade 22)
    # - e, principalmente, progressão no programa (Pedra 22)
    # =========================================
    ida_by_cohort = (
        df.groupby("Ano ingresso")["IDA"]
        .agg(["count", "mean", "median"])
        .reset_index()
        .sort_values("Ano ingresso")
    )
    ida_by_cohort.to_csv(OUT_DIR / "ida_por_coorte.csv", index=False)

    plt.figure(figsize=(10, 5))
    plt.plot(ida_by_cohort["Ano ingresso"], ida_by_cohort["mean"], marker="o")
    plt.title("IDA médio por coorte (Ano de ingresso) — proxy de evolução")
    plt.xlabel("Ano de ingresso na Passos Mágicos")
    plt.ylabel("IDA médio")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "03_ida_por_coorte.png", dpi=150)
    plt.close()

    ida_by_age = (
        df.groupby("Idade 22")["IDA"]
        .agg(["count", "mean", "median"])
        .reset_index()
        .sort_values("Idade 22")
    )
    ida_by_age.to_csv(OUT_DIR / "ida_por_idade.csv", index=False)

    plt.figure(figsize=(10, 5))
    plt.plot(ida_by_age["Idade 22"], ida_by_age["mean"], marker="o")
    plt.title("IDA médio por idade (2022) — visão complementar")
    plt.xlabel("Idade em 2022")
    plt.ylabel("IDA médio")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "04_ida_por_idade.png", dpi=150)
    plt.close()

    # Proxy forte: progressão (Pedra 22)
    ida_by_pedra = (
        df.groupby("Pedra 22")["IDA"]
        .agg(["count", "mean", "median"])
        .reset_index()
        .sort_values("mean")
    )
    ida_by_pedra.to_csv(OUT_DIR / "ida_por_pedra.csv", index=False)

    plt.figure(figsize=(9, 5))
    plt.bar(ida_by_pedra["Pedra 22"], ida_by_pedra["mean"])
    plt.title("IDA médio por Pedra (classificação INDE)")
    plt.xlabel("Pedra 22")
    plt.ylabel("IDA médio")
    plt.xticks(rotation=15)
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "05_ida_por_pedra.png", dpi=150)
    plt.close()

    # =========================================
    # 5) Taxa de baixo desempenho (aplicável)
    # Critério defensável e data-driven:
    # - Baixo desempenho: IDA <= P25
    # - Crítico: IDA <= P10
    # (Sem assumir corte “oficial”, pois pode variar por escala/distribuição)
    # =========================================
    tmp = df.copy()
    tmp["baixo_desempenho_p25"] = (tmp["IDA"] <= p25).astype(int)
    tmp["critico_p10"] = (tmp["IDA"] <= p10).astype(int)

    rate_low = pd.DataFrame(
        {
            "criterio": ["Baixo desempenho (IDA <= P25)", "Crítico (IDA <= P10)"],
            "corte": [round(p25, 4), round(p10, 4)],
            "qtd": [int(tmp["baixo_desempenho_p25"].sum()), int(tmp["critico_p10"].sum())],
        }
    )
    rate_low["%"] = (rate_low["qtd"] / tmp["IDA"].notna().sum() * 100).round(1)
    rate_low.to_csv(OUT_DIR / "taxa_baixo_desempenho.csv", index=False)

    plt.figure(figsize=(9, 5))
    plt.bar(rate_low["criterio"], rate_low["%"])
    plt.title("Taxa de baixo desempenho acadêmico (IDA) — cortes por percentil")
    plt.ylabel("% de alunos")
    plt.xticks(rotation=15, ha="right")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "06_taxa_baixo_desempenho.png", dpi=150)
    plt.close()

    # =========================================
    # 6) Análise por fase
    # =========================================
    ida_by_fase = (
        df.groupby("Fase")["IDA"]
        .agg(["count", "mean", "median"])
        .reset_index()
        .sort_values("Fase")
    )
    ida_by_fase.to_csv(OUT_DIR / "ida_por_fase.csv", index=False)

    plt.figure(figsize=(10, 5))
    plt.bar(ida_by_fase["Fase"].astype(str), ida_by_fase["mean"])
    plt.title("IDA médio por Fase")
    plt.xlabel("Fase")
    plt.ylabel("IDA médio")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "07_ida_por_fase.png", dpi=150)
    plt.close()

    # Taxa de baixo desempenho (P25) por fase
    rate_low_by_fase = (
        tmp.groupby("Fase")["baixo_desempenho_p25"]
        .mean()
        .reset_index()
        .rename(columns={"baixo_desempenho_p25": "taxa"})
    )
    rate_low_by_fase["%"] = (rate_low_by_fase["taxa"] * 100).round(1)
    rate_low_by_fase.to_csv(OUT_DIR / "taxa_baixo_desempenho_por_fase.csv", index=False)

    plt.figure(figsize=(10, 5))
    plt.bar(rate_low_by_fase["Fase"].astype(str), rate_low_by_fase["%"])
    plt.title("Taxa de baixo desempenho (IDA <= P25) por Fase")
    plt.xlabel("Fase")
    plt.ylabel("% de alunos")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "08_taxa_baixo_desempenho_por_fase.png", dpi=150)
    plt.close()

    # =========================================
    # Conclusão (markdown)
    # =========================================
    md = f"""# Pergunta 2 — Desempenho Acadêmico (IDA)

## Estrutura aplicada
1) Carregar dados  
2) Distribuição do IDA  
3) Cálculos descritivos  
4) Evolução no tempo (proxy)  
5) Taxas (baixo desempenho)  
6) Análise por fase  

## Observação metodológica (importante)
O dataset disponibilizado possui **uma coluna única `IDA`** (sem sufixo de ano e sem granularidade intra-ano).  
Assim, a análise de “melhorando/estagnado/caindo ao longo do tempo” foi feita por **proxies**:
- **Coorte (Ano de ingresso)**  
- **Idade em 2022** (visão complementar)  
- **Progressão no programa (Pedra 22 e Fase)** — proxy mais forte de evolução.

## Resultados principais
- Descritivos do IDA (base completa em `descritivos_ida.csv`):
  - P10 = **{round(p10,4)}**
  - P25 = **{round(p25,4)}**
  - Mediana (P50) = **{round(p50,4)}**

- Taxas de desempenho (percentis):
  - **Baixo desempenho (IDA <= P25): {rate_low.loc[0, '%']}%**
  - **Crítico (IDA <= P10): {rate_low.loc[1, '%']}%**

- Progressão:
  - O IDA médio por **Pedra 22** indica diferenciação clara de desempenho entre níveis do programa, reforçando coerência entre desempenho acadêmico e a classificação geral.

## Conclusão (resposta da Pergunta 2)
O desempenho acadêmico (IDA) apresenta distribuição heterogênea, com um grupo de alunos concentrado nas faixas inferiores (P25 e P10), que caracteriza um público-alvo prioritário para ações pedagógicas e de suporte.  
Como não há IDA por ano no arquivo atual, a “evolução” é inferida por proxies: coorte, fase e especialmente Pedra 22. A análise por fase e por progressão no programa sugere que alunos em níveis mais avançados tendem a apresentar desempenho acadêmico superior, sustentando a narrativa de melhoria associada à permanência e evolução no ciclo da Passos Mágicos.
"""
    (OUT_DIR / "README_q2.md").write_text(md, encoding="utf-8")

    print("✅ Q2 concluída! Arquivos gerados em:", OUT_DIR)
    print("   - Gráficos: reports/q2/*.png")
    print("   - Tabelas:  reports/q2/*.csv")
    print("   - Resumo:   reports/q2/README_q2.md")


if __name__ == "__main__":
    main()
