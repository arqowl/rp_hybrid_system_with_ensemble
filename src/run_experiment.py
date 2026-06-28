"""
run_experiment.py — Ponto de entrada por LINHA DE COMANDO (alternativa ao
notebook). Executa todo o pipeline ponta a ponta para Finnish e Maxwell,
reaproveitando as funcoes por fase de pipeline.py, e grava os artefatos:

  outputs/tables/tabela4_finnish.csv
  outputs/tables/tabela5_maxwell.csv
  outputs/tables/tabela6_wilcoxon.csv
  outputs/figures/fig4_finnish_scatter.png .. fig7_maxwell_radar.png
  outputs/report.md

Uso:  python src/run_experiment.py
(O mesmo experimento pode ser rodado, fase a fase, em
 notebooks/01_replicacao_OES.ipynb.)
"""
import os
import warnings
warnings.filterwarnings("ignore")
import pandas as pd

import pipeline as pl
import models as mdl
import figures as fg
from tabelas import (TARGET_T4, TARGET_T5, TARGET_T6, METRIC_COLS, ROW_ORDER,
                     montar_tabela_metricas, montar_tabela_wilcoxon,
                     radar_inputs)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TAB = os.path.join(ROOT, "outputs", "tables")
FIG = os.path.join(ROOT, "outputs", "figures")
os.makedirs(TAB, exist_ok=True)
os.makedirs(FIG, exist_ok=True)


def main():
    log = []
    res = {}
    for name in ["finnish", "maxwell"]:
        r = pl.rodar_base(name)
        res[name] = r
        d = r["d"]
        log.append(f"[{name}] checagem Tabela 2: "
                   f"{'OK' if d['check_ok'] else 'DIVERGENCIA'} | "
                   f"obs={d['check_got']['obs']} mean={d['check_got']['mean']:.4f} "
                   f"median={d['check_got']['median']} skew={d['check_got']['skew']:.2f} "
                   f"kurt={d['check_got']['kurt']:.2f}")
        log.append(f"[{name}] features={d['n_features']} (incl. alvo); "
                   f"treino={len(d['y_train'])} teste={len(d['y_test'])}")
        log.append(f"[{name}] SES-GA selecionou {len(r['sel_names'])} modelos: "
                   f"{r['sel_names']}")

    # Tabelas 4 e 5
    t4 = montar_tabela_metricas(res["finnish"]["rows"], TARGET_T4)
    t5 = montar_tabela_metricas(res["maxwell"]["rows"], TARGET_T5)
    t4.to_csv(os.path.join(TAB, "tabela4_finnish.csv"), index=False)
    t5.to_csv(os.path.join(TAB, "tabela5_maxwell.csv"), index=False)

    # Tabela 6
    t6 = montar_tabela_wilcoxon(res["finnish"]["wil"], res["maxwell"]["wil"])
    t6.to_csv(os.path.join(TAB, "tabela6_wilcoxon.csv"), index=False)

    # Figuras 4/5 (scatter) e 6/7 (radar)
    df_f = res["finnish"]["d"]; df_m = res["maxwell"]["d"]
    fg.scatter_fig(os.path.join(FIG, "fig4_finnish_scatter.png"),
                   "Figure 4. Scatter (Finnish)",
                   res["finnish"]["ens_train"], res["finnish"]["ens_test"],
                   df_f["y_train_raw"], df_f["y_test_raw"])
    fg.scatter_fig(os.path.join(FIG, "fig5_maxwell_scatter.png"),
                   "Figure 5. Scatter (Maxwell)",
                   res["maxwell"]["ens_train"], res["maxwell"]["ens_test"],
                   df_m["y_train_raw"], df_m["y_test_raw"])
    lab, s, c = radar_inputs(res["finnish"]["rows"])
    fg.radar_fig(os.path.join(FIG, "fig6_finnish_radar.png"),
                 "Figure 6. sMAPE vs COD (Finnish)", lab, s, c)
    lab, s, c = radar_inputs(res["maxwell"]["rows"])
    fg.radar_fig(os.path.join(FIG, "fig7_maxwell_radar.png"),
                 "Figure 7. sMAPE vs COD (Maxwell)", lab, s, c)

    # Relatorio
    from report import build_report
    build_report(ROOT, res, t4, t5, t6, log, METRIC_COLS, ROW_ORDER, mdl.POOL_ORDER)

    print("\n".join(log))
    print("\nArtefatos gravados em outputs/. OK.")


if __name__ == "__main__":
    main()
