"""
Pergunta 9 — ML: Previsão de risco de defasagem (AJUSTADO)
- Corrige erro do permutation_importance em matriz esparsa usando importância por coeficientes
- Gera outputs em reports/q9 e salva modelo/metadata em models/

Rodar:
    source .venv/bin/activate
    python scripts/q9_ml_risco_defasagem.py
"""

from __future__ import annotations

from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)
import joblib

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = REPO_ROOT / "data" / "raw" / "BASE_DE_DADOS_PEDE_2024_DATATHON.xlsx"
OUT_DIR = REPO_ROOT / "reports" / "q9"
MODEL_DIR = REPO_ROOT / "models"
OUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    # =========================================================
    # 1) Carregar dados
    # =========================================================
    df = pd.read_excel(DATA_PATH)

    required = [
        "Defas",
        "IDA",
        "IEG",
        "IAA",
        "IPS",
        "Fase",
        "Pedra 22",
        "Ano ingresso",
        "Idade 22",
        "Gênero",
        "Turma",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Colunas ausentes para Q9: {missing}. Colunas: {list(df.columns)}")

    # Tipos numéricos
    num_cols = ["IDA", "IEG", "IAA", "IPS", "Ano ingresso", "Idade 22"]
    df["Defas"] = pd.to_numeric(df["Defas"], errors="coerce")
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # =========================================================
    # 2) Definir alvo (RISCO) via Defas >= P75
    # =========================================================
    defas = df["Defas"].dropna()
    if defas.shape[0] < 50:
        raise ValueError("Poucos dados não-nulos em Defas para treinar modelo com segurança.")

    thr = float(defas.quantile(0.75))
    df["risco_defasagem"] = (df["Defas"] >= thr).astype(int)

    # Evitar vazamento: remover Defas e IAN (se existir)
    leak_cols = [c for c in ["Defas", "IAN"] if c in df.columns]
    df_model = df.drop(columns=leak_cols)

    # Features categóricas
    cat_cols = ["Fase", "Pedra 22", "Gênero", "Turma"]

    # Remover linhas sem alvo
    df_model = df_model.dropna(subset=["risco_defasagem"])

    X = df_model[num_cols + cat_cols]
    y = df_model["risco_defasagem"]

    # =========================================================
    # 3) Split treino/teste
    # =========================================================
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    # =========================================================
    # 4) Pipeline (preprocess + modelo)
    # =========================================================
    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    pre = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, num_cols),
            ("cat", categorical_pipe, cat_cols),
        ],
        remainder="drop",
    )

    clf = LogisticRegression(max_iter=2000, class_weight="balanced")
    pipe = Pipeline(steps=[("pre", pre), ("clf", clf)])

    # =========================================================
    # 5) Treinar
    # =========================================================
    pipe.fit(X_train, y_train)

    # Probabilidades
    p_test = pipe.predict_proba(X_test)[:, 1]

    # Métricas
    roc = roc_auc_score(y_test, p_test)
    pr_auc = average_precision_score(y_test, p_test)

    # Threshold default 0.5
    y_pred_05 = (p_test >= 0.5).astype(int)

    # =========================================================
    # 6) Outputs de avaliação
    # =========================================================
    report_txt = classification_report(y_test, y_pred_05, digits=3)
    (OUT_DIR / "classification_report.txt").write_text(report_txt, encoding="utf-8")

    cm = confusion_matrix(y_test, y_pred_05)
    pd.DataFrame(cm, index=["Real 0", "Real 1"], columns=["Pred 0", "Pred 1"]).to_csv(
        OUT_DIR / "confusion_matrix.csv", index=True
    )

    # ROC curve
    fpr, tpr, _ = roc_curve(y_test, p_test)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.title("ROC Curve — Risco de Defasagem")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "roc_curve.png", dpi=160)
    plt.close()

    # Precision-Recall curve
    prec, rec, _ = precision_recall_curve(y_test, p_test)
    plt.figure(figsize=(6, 5))
    plt.plot(rec, prec)
    plt.title("Precision-Recall Curve — Risco de Defasagem")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "pr_curve.png", dpi=160)
    plt.close()

    # =========================================================
    # 7) Importância de features (coeficientes do Logistic)
    #    - evita problema de matriz esparsa no permutation_importance
    # =========================================================
    # nomes das features após one-hot
    ohe = pipe.named_steps["pre"].named_transformers_["cat"].named_steps["onehot"]
    cat_feature_names = list(ohe.get_feature_names_out(cat_cols))
    feature_names = num_cols + cat_feature_names

    coefs = pipe.named_steps["clf"].coef_.ravel()

    imp = pd.DataFrame({"feature": feature_names, "coef": coefs})
    imp["abs_coef"] = imp["coef"].abs()
    imp = imp.sort_values("abs_coef", ascending=False)
    imp.head(30).to_csv(OUT_DIR / "feature_importance_top30.csv", index=False)

    # =========================================================
    # 8) Salvar modelo + metadata
    # =========================================================
    joblib.dump(pipe, MODEL_DIR / "q9_model.pkl")

    metadata = {
        "target": "risco_defasagem",
        "target_definition": "1 se Defas >= P75 (threshold calculado no dataset)",
        "defas_threshold_p75": thr,
        "n_rows_total": int(len(df)),
        "n_rows_model": int(len(df_model)),
        "positive_rate_model": float(y.mean()),
        "features_numeric": num_cols,
        "features_categorical": cat_cols,
        "roc_auc": float(roc),
        "pr_auc": float(pr_auc),
    }
    (MODEL_DIR / "q9_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    # =========================================================
    # 9) README automático
    # =========================================================
    md = f"""# Pergunta 9 — ML: Previsão de risco de defasagem

## Definição do alvo (risco)
- Risco = **Defas >= P75**
- Threshold (P75) = **{thr:.4f}**
- Taxa de risco no dataset (model) = **{y.mean():.3f}**

## Features usadas (sem vazamento)
Numéricas: {", ".join(num_cols)}
Categóricas: {", ".join(cat_cols)}
Removidas por vazamento: {", ".join(leak_cols) if leak_cols else "nenhuma"}

## Modelo
- Logistic Regression (class_weight=balanced) com pipeline de preprocessamento

## Métricas (teste)
- ROC AUC = **{roc:.4f}**
- PR AUC = **{pr_auc:.4f}**

Arquivos gerados:
- `classification_report.txt`
- `confusion_matrix.csv`
- `roc_curve.png`
- `pr_curve.png`
- `feature_importance_top30.csv`
- Modelo: `models/q9_model.pkl`
- Metadata: `models/q9_metadata.json`
"""
    (OUT_DIR / "README_q9.md").write_text(md, encoding="utf-8")

    print("✅ Q9 concluída! Outputs em:", OUT_DIR)
    print("   - Modelo salvo em: models/q9_model.pkl")
    print("   - Importância (coeficientes) em: reports/q9/feature_importance_top30.csv")


if __name__ == "__main__":
    main()