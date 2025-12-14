# Pergunta 3 — Engajamento nas atividades (IEG)

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
- P10 = **5.59**
- P25 = **7.0**
- Mediana (P50) = **8.3**
- P75 = **9.1**

## Taxas (engajamento)
- Baixo engajamento (IEG <= P25): **25.6%**
- Alto engajamento (IEG >= P75): **27.0%**

## Relação IEG ↔ IDA / IPV (core da Pergunta 3)
**Correlações (tabela completa em `correlacoes_ieg_ida_ipv.csv`):**
- IEG x IDA → Pearson: **0.5641** | Spearman: **0.5074**
- IEG x IPV → Pearson: **0.5892** | Spearman: **0.5403**

**Comparação Alto vs Baixo engajamento (tabela em `comparacao_alto_vs_baixo_engajamento.csv`):**
- IDA médio (Alto IEG) - (Baixo IEG) = **2.7074**
- IPV médio (Alto IEG) - (Baixo IEG) = **1.5442**

## Conclusão (resposta da Pergunta 3)
Os resultados mostram a distribuição do engajamento e permitem segmentar alunos em baixo e alto engajamento (P25/P75).
A análise de correlação e a comparação entre grupos (alto vs baixo IEG) indicam se existe uma relação consistente entre engajamento e desempenho acadêmico (IDA) e com o ponto de virada (IPV).
Caso as correlações sejam positivas e o grupo de alto engajamento apresente IDA/IPV médios maiores, isso sustenta a hipótese de que o engajamento é um fator-chave para evolução do aluno — e deve ser priorizado em intervenções.
