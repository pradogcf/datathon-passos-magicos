# Pergunta 6 — Aspectos Psicopedagógicos (IPP) [PROXY]

## Importante (limitação do dataset)
O dataset **não contém** a coluna numérica `IPP`.  
Para responder a dor, foi utilizado o campo **`Rec Psicologia`** como **proxy** psicopedagógico/psicológico.

## Cobertura do proxy
- Total de alunos: **860**
- Sem recomendação (nulos): **0** (**0.0%**)

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
