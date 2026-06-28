"""
pipeline.py — Funcoes por FASE do experimento (fonte unica de logica).

Tanto o notebook (notebooks/01_replicacao_OES.ipynb) quanto o script
(src/run_experiment.py) chamam estas funcoes — nenhuma logica e duplicada.
Cada funcao corresponde a uma fase do artigo e devolve os artefatos
intermediarios, para que o notebook possa exibi-los celula a celula.

Fluxo por base:
    d      = fase1_preprocess(name)
    pool   = fase2_treinar_pool(d)            # modelos individuais (Tabela 3)
    sel    = fase2_ses_ga(d, pool, rng)       # Static Ensemble Selection (GA)
    ens    = fase3_des_oes(d, pool, sel)      # DES + OES (Proposed)
    rows   = montar_linhas(d, pool, sel, ens) # 5 metricas por modelo
    wil    = wilcoxon_ensembles(d, ens)       # Tabela 6
"""
import numpy as np
from scipy.stats import wilcoxon
from sklearn.base import clone
from sklearn.model_selection import cross_val_predict, KFold

import data_prep as dp
import models as mdl
import ensembles as ens_mod
import metrics as mt

RANDOM_STATE = 42


# --- FASE 1 — Pre-processamento ------------------------------------------
def fase1_preprocess(name):
    """Carga + ausentes + checagem Tabela 2 + selecao 31/21 + MinMax + split."""
    return dp.build(name)


# --- FASE 2 — Pool de modelos (Tabela 3) ---------------------------------
def fase2_treinar_pool(d):
    """Treina cada modelo do pool. Devolve predicoes OOF (treino) e de teste,
    ja invertidas para o esforco BRUTO (regime do artigo), e as metricas
    individuais.

    oof : predicoes out-of-fold (5-fold) no treino -> usadas pela fitness do GA
          e pela competencia do DES (evita o vies de resubstituicao).
    """
    pool = mdl.build_pool()
    kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    yscl = d["y_scaler"]

    def inv(p):  # escala MinMax -> esforco bruto
        return yscl.inverse_transform(np.asarray(p).reshape(-1, 1)).ravel()

    oof_preds, test_preds, rows = [], [], {}
    for sig in mdl.POOL_ORDER:
        est = pool[sig]
        oof = inv(cross_val_predict(clone(est), d["X_train"], d["y_train"], cv=kf))
        est.fit(d["X_train"], d["y_train"])
        pte = inv(est.predict(d["X_test"]))
        oof_preds.append(oof)
        test_preds.append(pte)
        rows[sig] = mt.all_metrics(d["y_test_raw"], pte)

    return dict(oof_preds=np.array(oof_preds),
                test_preds=np.array(test_preds),
                rows_individuais=rows)


# --- FASE 2 — Static Ensemble Selection via GA ---------------------------
def fase2_ses_ga(d, pool, rng):
    """Roda o GA sobre as predicoes OOF e devolve a selecao + predicoes SES."""
    mask, sel_idx = ens_mod.ses_ga(pool["oof_preds"], d["y_train_raw"], rng)
    sel_names = [mdl.POOL_ORDER[i] for i in sel_idx]
    ses_tr = ens_mod._ensemble_pred(pool["oof_preds"], mask)
    ses_te = ens_mod._ensemble_pred(pool["test_preds"], mask)
    return dict(mask=mask, sel_idx=sel_idx, sel_names=sel_names,
                ses_tr=ses_tr, ses_te=ses_te)


# --- FASE 3 — Dynamic Ensemble Selection + Omni-Ensemble Selection -------
def fase3_des_oes(d, pool, sel):
    """DES (competencia local) e OES (combinacao SES+DES)."""
    des_tr = ens_mod.des_predict(sel["sel_idx"], pool["oof_preds"], pool["oof_preds"],
                                 d["X_train"], d["X_train"], d["y_train_raw"])
    des_te = ens_mod.des_predict(sel["sel_idx"], pool["oof_preds"], pool["test_preds"],
                                 d["X_train"], d["X_test"], d["y_train_raw"])
    oes_tr = ens_mod.oes_predict(sel["ses_tr"], des_tr)
    oes_te = ens_mod.oes_predict(sel["ses_te"], des_te)
    return dict(des_tr=des_tr, des_te=des_te, oes_tr=oes_tr, oes_te=oes_te)


# --- Metricas das 3 abordagens de ensemble + montagem das linhas ----------
def montar_linhas(d, pool, sel, ens):
    """Junta as metricas dos modelos individuais e dos 3 ensembles."""
    rows = dict(pool["rows_individuais"])
    rows["Static"] = mt.all_metrics(d["y_test_raw"], sel["ses_te"])
    rows["DES"] = mt.all_metrics(d["y_test_raw"], ens["des_te"])
    rows["Proposed"] = mt.all_metrics(d["y_test_raw"], ens["oes_te"])
    return rows


# --- Tabela 6 — Wilcoxon signed-rank (real vs predito, teste) -------------
def wilcoxon_ensembles(d, sel, ens):
    out = {}
    for key, pred in [("Static", sel["ses_te"]),
                      ("DES", ens["des_te"]),
                      ("Proposed", ens["oes_te"])]:
        try:
            out[key] = float(wilcoxon(d["y_test_raw"], pred).pvalue)
        except ValueError:
            out[key] = float("nan")
    return out


# --- Conveniencia: roda TODAS as fases de uma base ------------------------
def rodar_base(name):
    """Executa todas as fases de uma base e devolve um dict completo.

    Usado pelo run_experiment.py; o notebook chama as fases individualmente.
    """
    rng = np.random.default_rng(RANDOM_STATE)
    d = fase1_preprocess(name)
    pool = fase2_treinar_pool(d)
    sel = fase2_ses_ga(d, pool, rng)
    ens = fase3_des_oes(d, pool, sel)
    rows = montar_linhas(d, pool, sel, ens)
    wil = wilcoxon_ensembles(d, sel, ens)
    ens_train = {"SES": sel["ses_tr"], "DES": ens["des_tr"], "OES": ens["oes_tr"]}
    ens_test = {"SES": sel["ses_te"], "DES": ens["des_te"], "OES": ens["oes_te"]}
    return dict(d=d, rows=rows, wil=wil, sel_names=sel["sel_names"],
                ens_train=ens_train, ens_test=ens_test)
