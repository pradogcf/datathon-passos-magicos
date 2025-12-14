# Pergunta 2 — Desempenho Acadêmico (IDA)

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
  - P10 = **3.3**
  - P25 = **4.8**
  - Mediana (P50) = **6.3**

- Taxas de desempenho (percentis):
  - **Baixo desempenho (IDA <= P25): 26.5%**
  - **Crítico (IDA <= P10): 11.3%**

- Progressão:
  - O IDA médio por **Pedra 22** indica diferenciação clara de desempenho entre níveis do programa, reforçando coerência entre desempenho acadêmico e a classificação geral.

## Conclusão (resposta da Pergunta 2)
O desempenho acadêmico (IDA) apresenta distribuição heterogênea, com um grupo de alunos concentrado nas faixas inferiores (P25 e P10), que caracteriza um público-alvo prioritário para ações pedagógicas e de suporte.  
Como não há IDA por ano no arquivo atual, a “evolução” é inferida por proxies: coorte, fase e especialmente Pedra 22. A análise por fase e por progressão no programa sugere que alunos em níveis mais avançados tendem a apresentar desempenho acadêmico superior, sustentando a narrativa de melhoria associada à permanência e evolução no ciclo da Passos Mágicos.
