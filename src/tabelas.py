"""
tabelas.py — Valores-alvo do artigo (Tabelas 4, 5 e 6) e funcoes que montam as
tabelas comparativas (obtido x artigo x delta). Compartilhado pelo notebook e
pelo run_experiment.py.
"""
import pandas as pd
import models as mdl

METRIC_COLS = ["sMAPE", "MRE", "MASE", "NSE", "COD"]
ROW_ORDER = mdl.POOL_ORDER + ["Static", "DES", "Proposed"]

# Tabela 4 — Finnish (valores do artigo)
TARGET_T4 = {
    "SVM": [48.186, 1.87874, 0.81134, 0.56357, 0.80788],
    "RF": [29.145, 1.04612, 0.66909, 0.36538, 0.74756],
    "MLP": [38.703, 1.89333, 0.69446, 0.64448, 0.8107],
    "kNN": [35.681, 1.36927, 0.72336, 0.56898, 0.75544],
    "DT": [37.585, 2.25293, 0.92997, 0.28421, 0.62113],
    "ET": [27.747, 0.98134, 0.54617, 0.68537, 0.84353],
    "LR": [42.961, 1.45984, 0.67045, 0.6332, 0.84546],
    "ADA": [37.621, 2.46283, 0.80211, 0.23775, 0.72402],
    "CAT": [33.534, 1.44861, 0.69836, 0.50544, 0.79121],
    "XGB": [30.131, 1.34004, 0.69576, 0.2715, 0.77509],
    "NB": [38.732, 1.19796, 0.62284, 0.67665, 0.83747],
    "BG": [43.474, 1.54597, 0.70022, 0.62224, 0.8411],
    "Static": [27.825, 1.05944, 0.54115, 0.71622, 0.87652],
    "DES": [33.884, 0.92209, 0.5723, 0.66953, 0.84379],
    "Proposed": [23.896, 0.79142, 0.45167, 0.78121, 0.91375],
}
# Tabela 5 — Maxwell (valores do artigo)
TARGET_T5 = {
    "SVM": [37.302, 1.60133, 0.64518, 0.64141, 0.85525],
    "RF": [18.274, 0.69642, 0.32081, 0.88611, 0.95035],
    "MLP": [45.079, 2.61562, 0.98424, 0.12867, 0.2717],
    "kNN": [35.082, 1.4606, 0.78834, 0.43263, 0.73278],
    "DT": [30.156, 1.49436, 0.6474, 0.61278, 0.79335],
    "ET": [18.59, 1.1559, 0.37111, 0.805, 0.89854],
    "LR": [65.296, 14.70762, 1.61663, 0.28759, 0.63149],
    "ADA": [24.89, 0.96699, 0.57493, 0.43299, 0.95678],
    "CAT": [36.453, 1.54738, 0.673, 0.5657, 0.87383],
    "XGB": [18.637, 0.92803, 0.29177, 0.93225, 0.96763],
    "NB": [39.123, 1.0559, 0.57534, 0.66247, 0.81607],
    "BG": [20.589, 1.10704, 0.35486, 0.88439, 0.94103],
    "Static": [23.527, 0.7629, 0.42057, 0.8042, 0.93988],
    "DES": [16.558, 0.50287, 0.32678, 0.90453, 0.9522],
    "Proposed": [15.057, 0.47942, 0.26125, 0.95146, 0.98359],
}
# Tabela 6 — Wilcoxon (p-values do artigo)
TARGET_T6 = {
    "Static": {"finnish": 0.236, "maxwell": 0.112},
    "DES": {"finnish": 0.075, "maxwell": 0.056},
    "Proposed": {"finnish": 0.004, "maxwell": 0.021},
}


def montar_tabela_metricas(rows, target):
    """DataFrame por modelo: <metrica>_obt, _art e _delta para as 5 metricas."""
    recs = []
    for r in ROW_ORDER:
        got, tgt = rows[r], target[r]
        rec = {"Model": r}
        for j, m in enumerate(METRIC_COLS):
            rec[f"{m}_obt"] = round(got[m], 5)
            rec[f"{m}_art"] = tgt[j]
            rec[f"{m}_delta"] = round(got[m] - tgt[j], 5)
        recs.append(rec)
    return pd.DataFrame(recs)


def montar_tabela_wilcoxon(wil_f, wil_m):
    """Tabela 6: p-values obtidos x artigo, para SES/DES/OES."""
    nomes = [("Static", "Static Ensemble Selection (SES)"),
             ("DES", "Dynamic Ensemble Selection (DES)"),
             ("Proposed", "Omni Ensemble Selection (OES)")]
    recs = []
    for key, nome in nomes:
        recs.append({"Model": nome,
                     "Finnish_obt": round(wil_f[key], 4),
                     "Finnish_art": TARGET_T6[key]["finnish"],
                     "Maxwell_obt": round(wil_m[key], 4),
                     "Maxwell_art": TARGET_T6[key]["maxwell"]})
    return pd.DataFrame(recs)


def radar_inputs(rows):
    """Rotulos e series (sMAPE, COD) para o radar das Figuras 6/7."""
    order = ROW_ORDER  # 12 individuais + Static, DES, Proposed
    labels = mdl.POOL_ORDER + ["SES", "DES", "OES"]
    s = [rows[o]["sMAPE"] for o in order]
    c = [rows[o]["COD"] for o in order]
    return labels, s, c
