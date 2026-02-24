# Pergunta 9 — ML: Previsão de risco de defasagem

## Definição do alvo (risco)
- 1 se Defas > 0 (qualquer defasagem)
- Referência/threshold usado = **0.0000**
- Taxa de risco no dataset (model) = **0.014**

## Features usadas (sem vazamento)
Numéricas: IDA, IEG, IAA, IPS, Ano ingresso, Idade 22
Categóricas: Fase, Pedra 22, Gênero, Turma
Removidas por vazamento: Defas, IAN

## Modelo
- Logistic Regression (class_weight=balanced) com pipeline de preprocessamento

## Métricas (teste)
- ROC AUC = **0.8962**
- PR AUC = **0.1339**

Arquivos gerados:
- `classification_report.txt`
- `confusion_matrix.csv`
- `roc_curve.png`
- `pr_curve.png`
- `feature_importance_top30.csv`
- Modelo: `models/q9_model.pkl`
- Metadata: `models/q9_metadata.json`
