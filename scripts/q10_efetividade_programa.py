"""
Pergunta 10 — Efetividade do Programa

Objetivo:
Verificar se os indicadores melhoram conforme o aluno avança nas fases
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/raw/BASE_DE_DADOS_PEDE_2024_DATATHON.xlsx"
OUT = ROOT / "reports/q10"
OUT.mkdir(parents=True, exist_ok=True)

def main():

    df = pd.read_excel(DATA)

    cols = ["Fase","INDE 22","IDA","IEG","IPV","IAN"]
    df = df[cols]

    for c in cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # =====================================================
    # 1 — Médias por fase
    # =====================================================
    fase = df.groupby("Fase").mean().reset_index().sort_values("Fase")
    fase.to_csv(OUT/"medias_por_fase.csv", index=False)

    # =====================================================
    # 2 — Gráficos evolução
    # =====================================================
    indicadores = ["INDE 22","IDA","IEG","IPV","IAN"]

    for ind in indicadores:
        plt.figure()
        plt.plot(fase["Fase"], fase[ind], marker="o")
        plt.title(f"Evolução do {ind} por Fase")
        plt.xlabel("Fase")
        plt.ylabel(ind)
        plt.savefig(OUT/f"{ind}_evolucao.png")
        plt.close()

    # =====================================================
    # 3 — Crescimento total
    # =====================================================
    crescimento = []

    for ind in indicadores:
        crescimento.append([
            ind,
            fase[ind].iloc[-1] - fase[ind].iloc[0]
        ])

    crescimento = pd.DataFrame(
        crescimento,
        columns=["Indicador","Crescimento Total"]
    )

    crescimento.to_csv(OUT/"crescimento_total.csv", index=False)

    # =====================================================
    # 4 — README automático
    # =====================================================
    md = f"""
# Pergunta 10 — Efetividade do Programa

## Médias por fase
{fase.to_string(index=False)}

## Crescimento ao longo do programa
{crescimento.to_string(index=False)}

## Conclusão
Se os indicadores apresentam crescimento positivo ao longo das fases,
isso confirma a efetividade do programa na evolução dos alunos.
"""

    (OUT/"README_q10.md").write_text(md)

    print("Q10 concluída!")

if __name__ == "__main__":
    main()