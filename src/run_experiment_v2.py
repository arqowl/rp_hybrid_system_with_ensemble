"""
run_experiment_v2.py — Roda o experimento com o PRE-PROCESSAMENTO MELHORADO
(data_prep_v2) reaproveitando exatamente as mesmas fases de pipeline.py
(pool -> SES-GA -> DES -> OES, 5 metricas, Wilcoxon).

Gera, em outputs/v2/:
  tables/tabela4_finnish_v2.csv, tabela5_maxwell_v2.csv  (v2 x v1 x artigo)
  tables/tabela6_wilcoxon_v2.csv                         (v2 x v1 x artigo)
  tables/robustez_cv.csv                                 (CV repetida: media +- dp)
  figures/fig4..fig7                                     (scatter/radar v2)
  report_v2.md                                           (comparacao e discussao)

Uso:  python src/run_experiment_v2.py
"""
import os
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

import data_prep_v2 as dpv
import pipeline as pl
import models as mdl
import figures as fg
import tabelas as tb

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
V2 = os.path.join(ROOT, "outputs", "v2")
V2TAB = os.path.join(V2, "tables"); V2FIG = os.path.join(V2, "figures")
os.makedirs(V2TAB, exist_ok=True); os.makedirs(V2FIG, exist_ok=True)
V1TAB = os.path.join(ROOT, "outputs", "tables")   # resultados da replicacao fiel

N_REPEATS = 10        # CV repetida (splits estratificados) para estabilidade
METRIC_COLS = tb.METRIC_COLS
ROW_ORDER = tb.ROW_ORDER


# --------------------------------------------------------------------------
def rodar_v2(name, split_seed=42):
    """Roda todas as fases (pipeline.py) sobre o pre-processamento v2."""
    rng = np.random.default_rng(split_seed)
    d = dpv.build(name, split_seed=split_seed)
    pool = pl.fase2_treinar_pool(d)
    sel = pl.fase2_ses_ga(d, pool, rng)
    ens = pl.fase3_des_oes(d, pool, sel)
    rows = pl.montar_linhas(d, pool, sel, ens)
    wil = pl.wilcoxon_ensembles(d, sel, ens)
    ens_train = {"SES": sel["ses_tr"], "DES": ens["des_tr"], "OES": ens["oes_tr"]}
    ens_test = {"SES": sel["ses_te"], "DES": ens["des_te"], "OES": ens["oes_te"]}
    return dict(d=d, rows=rows, wil=wil, sel_names=sel["sel_names"],
                ens_train=ens_train, ens_test=ens_test)


def carregar_v1(csv):
    """Le os valores obtidos na replicacao fiel (colunas *_obt)."""
    df = pd.read_csv(os.path.join(V1TAB, csv))
    return {r["Model"]: {m: r[f"{m}_obt"] for m in METRIC_COLS}
            for _, r in df.iterrows()}


def tabela_comparativa(rows_v2, v1, target):
    """Por modelo e metrica: v2_obt, v1_obt, delta(v2-v1) e artigo."""
    recs = []
    for r in ROW_ORDER:
        rec = {"Model": r}
        for j, m in enumerate(METRIC_COLS):
            v2v = round(rows_v2[r][m], 4)
            v1v = round(float(v1[r][m]), 4)
            rec[f"{m}_v2"] = v2v
            rec[f"{m}_v1"] = v1v
            rec[f"{m}_delta"] = round(v2v - v1v, 4)
            rec[f"{m}_art"] = target[r][j]
        recs.append(rec)
    return pd.DataFrame(recs)


def robustez_cv(name):
    """CV repetida via N_REPEATS splits estratificados; media +- dp das
    metricas para os 3 ensembles. Estabiliza a estimativa (esp. Maxwell)."""
    acc = {k: {m: [] for m in METRIC_COLS} for k in ["Static", "DES", "Proposed"]}
    for s in range(N_REPEATS):
        r = rodar_v2(name, split_seed=100 + s)
        for k in acc:
            for m in METRIC_COLS:
                acc[k][m].append(r["rows"][k][m])
    recs = []
    nomes = {"Static": "SES", "DES": "DES", "Proposed": "OES"}
    for k in ["Static", "DES", "Proposed"]:
        rec = {"Ensemble": nomes[k]}
        for m in METRIC_COLS:
            a = np.array(acc[k][m])
            rec[f"{m}_mean"] = round(float(a.mean()), 4)
            rec[f"{m}_std"] = round(float(a.std()), 4)
        recs.append(rec)
    return pd.DataFrame(recs)


# --------------------------------------------------------------------------
def main():
    print("Rodando v2 (pre-processamento melhorado) ...")
    res = {n: rodar_v2(n, split_seed=42) for n in ["finnish", "maxwell"]}
    for n in res:
        d = res[n]["d"]
        print(f"  [{n}] features={d['n_features']} (incl. alvo) | "
              f"treino={len(d['y_train'])} teste={len(d['y_test'])} | "
              f"SES-GA={res[n]['sel_names']}")

    v1_f = carregar_v1("tabela4_finnish.csv")
    v1_m = carregar_v1("tabela5_maxwell.csv")
    t4 = tabela_comparativa(res["finnish"]["rows"], v1_f, tb.TARGET_T4)
    t5 = tabela_comparativa(res["maxwell"]["rows"], v1_m, tb.TARGET_T5)
    t4.to_csv(os.path.join(V2TAB, "tabela4_finnish_v2.csv"), index=False)
    t5.to_csv(os.path.join(V2TAB, "tabela5_maxwell_v2.csv"), index=False)

    # Tabela 6 — Wilcoxon v2 x v1
    v1w_f = pd.read_csv(os.path.join(V1TAB, "tabela6_wilcoxon.csv"))
    v1w_m = v1w_f.set_index("Model")["Maxwell_obt"].to_dict()
    v1w_fd = v1w_f.set_index("Model")["Finnish_obt"].to_dict()
    nomes = [("Static", "Static Ensemble Selection (SES)"),
             ("DES", "Dynamic Ensemble Selection (DES)"),
             ("Proposed", "Omni Ensemble Selection (OES)")]
    t6 = pd.DataFrame([{
        "Model": nome,
        "Finnish_v2": round(res["finnish"]["wil"][k], 4),
        "Finnish_v1": round(float(v1w_fd[nome]), 4),
        "Maxwell_v2": round(res["maxwell"]["wil"][k], 4),
        "Maxwell_v1": round(float(v1w_m[nome]), 4),
    } for k, nome in nomes])
    t6.to_csv(os.path.join(V2TAB, "tabela6_wilcoxon_v2.csv"), index=False)

    # Figuras v2
    df_f = res["finnish"]["d"]; df_m = res["maxwell"]["d"]
    fg.scatter_fig(os.path.join(V2FIG, "fig4_finnish_scatter.png"),
                   "Figure 4 (v2). Scatter (Finnish)",
                   res["finnish"]["ens_train"], res["finnish"]["ens_test"],
                   df_f["y_train_raw"], df_f["y_test_raw"])
    fg.scatter_fig(os.path.join(V2FIG, "fig5_maxwell_scatter.png"),
                   "Figure 5 (v2). Scatter (Maxwell)",
                   res["maxwell"]["ens_train"], res["maxwell"]["ens_test"],
                   df_m["y_train_raw"], df_m["y_test_raw"])
    lab, s, c = tb.radar_inputs(res["finnish"]["rows"])
    fg.radar_fig(os.path.join(V2FIG, "fig6_finnish_radar.png"),
                 "Figure 6 (v2). sMAPE vs COD (Finnish)", lab, s, c)
    lab, s, c = tb.radar_inputs(res["maxwell"]["rows"])
    fg.radar_fig(os.path.join(V2FIG, "fig7_maxwell_radar.png"),
                 "Figure 7 (v2). sMAPE vs COD (Maxwell)", lab, s, c)

    # Robustez (CV repetida)
    print(f"Rodando CV repetida ({N_REPEATS} splits estratificados por base) ...")
    rob_f = robustez_cv("finnish"); rob_f.insert(0, "Dataset", "Finnish")
    rob_m = robustez_cv("maxwell"); rob_m.insert(0, "Dataset", "Maxwell")
    rob = pd.concat([rob_f, rob_m], ignore_index=True)
    rob.to_csv(os.path.join(V2TAB, "robustez_cv.csv"), index=False)

    escrever_report(res, t4, t5, t6, rob)
    print("\nArtefatos v2 gravados em outputs/v2/. OK.")


def _cmp_ensembles_md(t, metric_cols):
    """Bloco markdown: so as 3 linhas de ensemble, v2 x v1 x delta x artigo."""
    sub = t[t["Model"].isin(["Static", "DES", "Proposed"])]
    head = "| Modelo | " + " | ".join(f"{m} v2 | {m} v1 | Δ | art" for m in metric_cols) + " |"
    sep = "|" + "---|" * (1 + 4 * len(metric_cols))
    lines = [head, sep]
    rename = {"Static": "SES", "DES": "DES", "Proposed": "OES"}
    for _, r in sub.iterrows():
        cells = [rename[r["Model"]]]
        for m in metric_cols:
            cells += [f"{r[f'{m}_v2']}", f"{r[f'{m}_v1']}", f"{r[f'{m}_delta']}", f"{r[f'{m}_art']}"]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def escrever_report(res, t4, t5, t6, rob):
    fin_drop = res["finnish"]["d"]["drop"]
    mx_drop = res["maxwell"]["d"]["drop"]

    rob_md = ["| Dataset | Ensemble | " + " | ".join(f"{m} (média±dp)" for m in METRIC_COLS) + " |",
              "|" + "---|" * (2 + len(METRIC_COLS))]
    for _, r in rob.iterrows():
        cells = [r["Dataset"], r["Ensemble"]]
        for m in METRIC_COLS:
            cells.append(f"{r[f'{m}_mean']} ± {r[f'{m}_std']}")
        rob_md.append("| " + " | ".join(cells) + " |")
    rob_md = "\n".join(rob_md)

    t6_md = ["| Modelo | Finnish v2 | Finnish v1 | Maxwell v2 | Maxwell v1 |",
             "|---|---|---|---|---|"]
    for _, r in t6.iterrows():
        t6_md.append(f"| {r['Model']} | {r['Finnish_v2']} | {r['Finnish_v1']} | "
                     f"{r['Maxwell_v2']} | {r['Maxwell_v1']} |")
    t6_md = "\n".join(t6_md)

    txt = f"""# Relatório v2 — Pré-processamento melhorado (comparação com a replicação fiel)

Esta variante mantém **a mesma arquitetura** (pool da Tabela 3 → SES-GA com
fitness de acurácia+diversidade → DES → OES → 5 métricas → Wilcoxon) e muda
**apenas o pré-processamento**, para isolar o efeito dos dados. Todas as
métricas seguem calculadas no **esforço bruto** (o alvo é log-transformado só
para o treino e revertido na avaliação), então os números são comparáveis com a
execução anterior (colunas `v1`).

## O que mudou no pré-processamento (3 melhorias)

1. **Alvo em log1p (+MinMax)** nas duas bases — corrige a forte assimetria
   (skew 3.7/3.3 → ~0). Treina-se no espaço log; as métricas são revertidas
   para o esforço bruto.
2. **Remoção direcionada de atributos** (além do recorte da replicação):
   - Finnish (removidos a mais): `AllFP_ep20` (agregado = soma das componentes,
     VIF = ∞), `IntFP`, `AlgFP` (variância quase-nula) → conjunto final:
     {res['finnish']['d']['n_features']} features (incl. alvo).
   - Maxwell (removidos a mais): `Har`, `T01`, `T04`, `T06`, `T13` (informação
     mútua ≈ 0) → conjunto final: {res['maxwell']['d']['n_features']} features
     (incl. alvo).
3. **Split 70:30 estratificado por quantis do alvo** (em vez de aleatório puro)
   + **validação cruzada repetida** ({N_REPEATS} splits) para estimativas estáveis.

> Não foram aplicados **PCA** nem **resampling de regressão**: o diagnóstico
> mostrou baixa compressibilidade (95% da variância exige 23/30 e 16/20
> componentes), tornando-os de baixo valor aqui. Podem ser adicionados se
> desejado.

## Tabela 4 — Finnish: ensembles v2 × v1 × artigo

{_cmp_ensembles_md(t4, METRIC_COLS)}

(Comparação completa de todos os modelos em `outputs/v2/tables/tabela4_finnish_v2.csv`.)

## Tabela 5 — Maxwell: ensembles v2 × v1 × artigo

{_cmp_ensembles_md(t5, METRIC_COLS)}

(Comparação completa em `outputs/v2/tables/tabela5_maxwell_v2.csv`.)

## Tabela 6 — Wilcoxon (p-value): v2 × v1

{t6_md}

## Robustez — validação cruzada repetida ({N_REPEATS} splits estratificados)

Média ± desvio-padrão das métricas dos 3 ensembles (esforço bruto). O desvio
indica o quanto o resultado de um único holdout é confiável — relevante
sobretudo no Maxwell (teste ~19 pontos).

{rob_md}

## Como ler a comparação

- **NSE/COD**: maior é melhor. **sMAPE/MRE/MASE**: menor é melhor.
- A coluna **Δ** é (v2 − v1) no esforço bruto. Δ negativo em sMAPE/MRE/MASE e
  positivo em NSE/COD indicam **melhora** com o pré-processamento.
- A coluna **art** repete o valor do artigo (referência), lembrando que a
  igualdade decimal não é esperada (features exatas e hiperparâmetros do GA/DES
  não constam do artigo).

Arquivos: tabelas em `outputs/v2/tables/`, figuras em `outputs/v2/figures/`.
"""
    path = os.path.join(V2, "report_v2.md")
    with open(path, "w") as f:
        f.write(txt)
    return path


if __name__ == "__main__":
    main()
