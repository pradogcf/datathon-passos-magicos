"""
Pergunta 6 — Aspectos Psicopedagógicos (IPP) [PROXY]
Dataset não possui coluna numérica 'IPP'. Usaremos 'Rec Psicologia' como proxy.

Objetivo: verificar se recomendações psicopedagógicas/psicológicas confirmam
ou contradizem a defasagem identificada pelo IAN/Defas.

Gera:
- reports/q6/*.png
- reports/q6/*.csv
- reports/q6/README_q6.md

Rodar:
    python3 scripts/q6_ipp_proxy_recpsicologia.py
"""

from __future__ import annotations

from pathlib import Path
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = REPO_ROOT / "data" / "raw" / "BASE_DE_DADOS_PEDE_2024_DATATHON.xlsx"
OUT_DIR = REPO_ROOT / "reports" / "q6"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def normalize_text(x: str) -> str:
    if not isinstance(x, str):
        return ""
    x = x.strip().lower()
    x = re.sub(r"\s+", " ", x)
    return x


def bucket_rec(text: str) -> str:
    """
    Heurística simples e defensável para agrupar recomendações.
    Ajustável depois se você quiser.
    """
    t = normalize_text(text)
    if t == "" or t in {"nan", "none"}:
        return "Sem recomendação"

    # Palavras-chave comuns (podem variar; o script também exporta top termos)
    if any(k in t for k in ["encaminhar", "encaminhamento", "psicólogo", "psicologia", "terapia", "atendimento"]):
        return "Encaminhamento/Acompanhamento"
    if any(k in t for k in ["atenção", "foco", "concentra", "concentração", "agitação", "hiper"]):
        return "Atenção/Concentração"
    if any(k in t for k in ["ans", "ansiedade", "depress", "triste", "humor", "estresse", "stress"]):
        return "Saúde emocional"
    if any(k in t for k in ["fam", "família", "responsável", "casa", "contexto"]):
        return "Família/Contexto"
    if any(k in t for k in ["autonomia", "organiza", "rotina", "disciplina"]):
        return "Rotina/Autonomia"

    return "Outros"


def main() -> None:
    # 1) Carregar dados
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Não encontrei o dataset em: {DATA_PATH}")

    df = pd.read_excel(DATA_PATH)

    required = ["Rec Psicologia", "IAN", "Defas", "Fase", "Pedra 22", "IDA", "IEG"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Colunas esperadas ausentes: {missing}. Colunas disponíveis: {list(df.columns)}")

    # Tipagem numérica
    for c in ["IAN", "Defas", "IDA", "IEG"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # 2) Distribuição (frequências)
    rec = df["Rec Psicologia"]
    n_total = len(df)
    n_missing = int(rec.isna().sum())
    pct_missing = round(n_missing / n_total * 100, 1)

    # Normaliza / bucketiza
    df["rec_psico_norm"] = rec.map(normalize_text)
    df["rec_psico_bucket"] = df["rec_psico_norm"].map(bucket_rec)

    freq_bucket = (
        df["rec_psico_bucket"]
        .value_counts(dropna=False)
        .rename_axis("categoria")
        .reset_index(name="qtd")
    )
    freq_bucket["%"] = (freq_bucket["qtd"] / n_total * 100).round(1)
    freq_bucket.to_csv(OUT_DIR / "freq_rec_psicologia_bucket.csv", index=False)

    plt.figure(figsize=(10, 5))
    plt.bar(freq_bucket["categoria"], freq_bucket["qtd"])
    plt.title("Rec Psicologia (proxy IPP) — frequência por categoria")
    plt.xlabel("Categoria")
    plt.ylabel("Quantidade de alunos")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "01_freq_rec_psicologia_bucket.png", dpi=150)
    plt.close()

    # 3) “Descritivos” textuais: top termos
    # Extrai palavras (simples)
    words = []
    for t in df["rec_psico_norm"].dropna():
        if not t:
            continue
        tokens = re.findall(r"[a-zà-ú]+", t)
        words.extend(tokens)

    stop = set([
        "de","da","do","das","dos","a","o","as","os","e","em","para","por","com","sem",
        "que","na","no","nas","nos","um","uma","ao","à","às","é","ser","foi","está","estao",
        "aluno","aluna","criança","crianca","jovem"
    ])
    words = [w for w in words if w not in stop and len(w) >= 3]
    top_terms = pd.Series(words).value_counts().head(30).rename_axis("termo").reset_index(name="freq")
    top_terms.to_csv(OUT_DIR / "top_termos_rec_psicologia.csv", index=False)

    # 4) Evolução proxy (por fase/pedra)
    by_fase = df.groupby("Fase")["rec_psico_bucket"].value_counts(normalize=True).rename("pct").reset_index()
    by_fase["pct"] = (by_fase["pct"] * 100).round(1)
    by_fase.to_csv(OUT_DIR / "rec_bucket_por_fase_pct.csv", index=False)

    # 5) Confirma ou contradiz? (relação com Defas / IAN)
    # Métricas por categoria: IAN médio, Defas médio, taxa de defasagem moderada/severa (se aplicável)
    # No dataset, Defas parece numérico; vamos criar cortes robustos por percentis
    defas = df["Defas"].dropna()
    if defas.shape[0] >= 30:
        d75 = float(defas.quantile(0.75))
        d90 = float(defas.quantile(0.90))
    else:
        d75, d90 = float("nan"), float("nan")

    df["defas_alta_p75"] = (df["Defas"] >= d75).astype(int) if np.isfinite(d75) else np.nan
    df["defas_muito_alta_p90"] = (df["Defas"] >= d90).astype(int) if np.isfinite(d90) else np.nan

    metrics = df.groupby("rec_psico_bucket").agg(
        n=("RA", "count") if "RA" in df.columns else ("IAN", "count"),
        IAN_medio=("IAN", "mean"),
        Defas_media=("Defas", "mean"),
        IDA_medio=("IDA", "mean"),
        IEG_medio=("IEG", "mean"),
        taxa_defas_alta_p75=("defas_alta_p75", "mean"),
        taxa_defas_muito_alta_p90=("defas_muito_alta_p90", "mean"),
    ).reset_index()

    metrics["taxa_defas_alta_p75"] = (metrics["taxa_defas_alta_p75"] * 100).round(1)
    metrics["taxa_defas_muito_alta_p90"] = (metrics["taxa_defas_muito_alta_p90"] * 100).round(1)
    metrics.to_csv(OUT_DIR / "metricas_por_categoria_rec_psicologia.csv", index=False)

    # Gráfico: Defas média por categoria
    m2 = metrics.sort_values("Defas_media")
    plt.figure(figsize=(10, 5))
    plt.bar(m2["rec_psico_bucket"], m2["Defas_media"])
    plt.title("Defasagem (Defas) média por categoria de Rec Psicologia")
    plt.xlabel("Categoria (Rec Psicologia)")
    plt.ylabel("Defas média")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "02_defas_media_por_categoria.png", dpi=150)
    plt.close()

    # 6) README / Conclusão
    md = f"""# Pergunta 6 — Aspectos Psicopedagógicos (IPP) [PROXY]

## Importante (limitação do dataset)
O dataset **não contém** a coluna numérica `IPP`.  
Para responder a dor, foi utilizado o campo **`Rec Psicologia`** como **proxy** psicopedagógico/psicológico.

## Cobertura do proxy
- Total de alunos: **{n_total}**
- Sem recomendação (nulos): **{n_missing}** (**{pct_missing}%**)

## Categorias (bucketização)
Tabela completa: `freq_rec_psicologia_bucket.csv`

## Validação da defasagem (IAN/Defas)
Avaliamos se categorias de recomendação se associam a:
- **IAN médio** (adequação)
- **Defas média** (defasagem)
- **Taxa de defasagem alta (>= P75)**
- **Taxa de defasagem muito alta (>= P90)**

Tabela completa: `metricas_por_categoria_rec_psicologia.csv`

## Conclusão (resposta da Pergunta 6)
Se categorias com **maior necessidade de acompanhamento** apresentarem **Defas média maior / IAN médio menor** e maior taxa de defasagem,
isso indica que as avaliações/recomendações psicopedagógicas **confirmam** a defasagem observada pelo IAN.
Se não houver diferença, indica baixa relação direta (ou necessidade de refinamento do indicador/registro).
"""
    (OUT_DIR / "README_q6.md").write_text(md, encoding="utf-8")

    print("✅ Q6 concluída! Arquivos gerados em:", OUT_DIR)
    print("   - Gráficos: reports/q6/*.png")
    print("   - Tabelas:  reports/q6/*.csv")
    print("   - Resumo:   reports/q6/README_q6.md")


if __name__ == "__main__":
    main()