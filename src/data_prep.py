"""
data_prep.py — Fase 1 (Pre-processamento) do artigo Jadhav et al. (2023).

Para cada base (Finnish, Maxwell):
  1. Importa o arquivo bruto de data/raw/.
  2. Trata ausentes (Finnish: remove 2 obs. invalidas -> 405 linhas).
  3. Valida estatisticas do alvo contra a Tabela 2 do artigo.
  4. Seleciona features (Finnish -> 31 incl. alvo; Maxwell -> 21 incl. alvo).
  5. Normalizacao MinMax [0,1] nas features (Fig. 3: normaliza ANTES do split).
  6. Holdout 70:30 com random_state fixo.

As metricas usadas sao invariantes a escala, entao mantemos o ALVO em valores
brutos (nao escalado) por interpretabilidade; apenas X e escalado.
"""
import os
import numpy as np
import pandas as pd
from scipy.io import arff
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

RANDOM_STATE = 42
TEST_SIZE = 0.30

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "data", "raw")

# Estatisticas de referencia do alvo (Tabela 2 do artigo) para checagem.
TABELA2 = {
    "finnish": dict(obs=405, mean=5031.0148, median=2500, mn=55, mx=63694,
                    skew=3.70, kurt=18.69),
    "maxwell": dict(obs=62, mean=8223.2097, median=5189.5, mn=583, mx=63694,
                    skew=3.34, kurt=13.69),
}

# ---------------------------------------------------------------------------
# Selecao de features (documentada em 'Pressupostos' do relatorio)
# ---------------------------------------------------------------------------
# O artigo nomeia 14 atributos "soft" de impacto na ISS, mas reporta um total
# de 31 (Finnish) e 21 (Maxwell) features incluindo o alvo. Os 14 nomes nao
# fecham 31/21; o artigo nao lista as colunas exatas. Adotamos uma regra
# simples e defensavel que atinge EXATAMENTE 31 e 21, descartando colunas
# nao-preditivas/identificadoras/de vazamento, e documentamos.

# FINNISH (46 -> 31): descarta 15 colunas =
#   7 identificadores/texto nominal + Duration (vazamento a priori)
#   + 7 contagens brutas "Tot" (redundantes com suas versoes "FP").
FINNISH_DROP = [
    "Case_Number", "Project_tech_ID", "YK", "Project_name",
    "Business_names", "Protype_names", "Hardware_names",   # 7 id/texto
    "Duration",                                            # vazamento a priori
    "InpTot", "InqTot", "OutTot", "IntTot",                # 7 contagens brutas
    "EntTot", "AlgTot", "AllTot",
]

# MAXWELL (27 -> 21): descarta 6 colunas =
#   Duration, Time (tempo decorrido, so conhecido ao fim -> vazamento)
#   + Syear, Source, Telonuse, Nlan (metadados de calendario/origem,
#     menor impacto direto no esforco). Mantem os 15 fatores T01..T15,
#   Size, Effort(alvo) e o contexto de sistema App/Har/Dba/Ifc.
MAXWELL_DROP = ["Duration", "Time", "Syear", "Source", "Telonuse", "Nlan"]

CONFIG = {
    "finnish": dict(fmt="csv", file="Finnish407.csv", target="Worksup",
                    drop=FINNISH_DROP, n_features=31),
    "maxwell": dict(fmt="arff", file="maxwell.arff", target="Effort",
                    drop=MAXWELL_DROP, n_features=21),
}


def _load_raw(cfg):
    path = os.path.join(RAW, cfg["file"])
    if cfg["fmt"] == "csv":
        return pd.read_csv(path)
    data, _ = arff.loadarff(path)
    df = pd.DataFrame(data)
    for c in df.columns:                       # ARFF traz bytes -> numerico
        if df[c].dtype == object:
            df[c] = pd.to_numeric(df[c].apply(
                lambda v: v.decode() if isinstance(v, bytes) else v),
                errors="coerce")
    return df


def validar_tabela2(name, y):
    """Recalcula stats do alvo e compara com a Tabela 2 (tolerancia p/ arred.)."""
    ref = TABELA2[name]
    got = dict(obs=len(y), mean=float(np.mean(y)), median=float(np.median(y)),
               mn=float(np.min(y)), mx=float(np.max(y)),
               skew=float(pd.Series(y).skew()),
               kurt=float(pd.Series(y).kurt()))
    ok = (got["obs"] == ref["obs"]
          and abs(got["mean"] - ref["mean"]) < 0.5
          and abs(got["median"] - ref["median"]) < 0.5
          and got["mn"] == ref["mn"] and got["mx"] == ref["mx"]
          and abs(got["skew"] - ref["skew"]) < 0.05
          and abs(got["kurt"] - ref["kurt"]) < 0.1)
    return ok, got, ref


def build(name):
    """Retorna dict com X/y de treino e teste + metadados da base."""
    cfg = CONFIG[name]
    df = _load_raw(cfg)
    target = cfg["target"]

    # Passo 2 — ausentes. Finnish: 2 obs. com esforco invalido (==0) sao as
    # "missing values" do artigo; remove-las reproduz exatamente a Tabela 2.
    if name == "finnish":
        df = df[df[target] > 0].reset_index(drop=True)
    df = df.dropna(subset=[target]).reset_index(drop=True)

    # Passo 3 — checagem Tabela 2 (no alvo, antes de qualquer escala).
    ok, got, ref = validar_tabela2(name, df[target].values)

    # Passo 4 — selecao de features (atinge exatamente n_features incl. alvo).
    keep = [c for c in df.columns if c not in cfg["drop"]]
    assert target in keep, "alvo nao pode ser descartado"
    sel = df[keep].copy()
    assert sel.shape[1] == cfg["n_features"], \
        f"{name}: {sel.shape[1]} features, esperado {cfg['n_features']}"

    feat_cols = [c for c in keep if c != target]
    X = sel[feat_cols].astype(float).values
    y_raw = sel[target].astype(float).values

    # Passo 5 — MinMax (Fig.3: normaliza features incl. alvo, antes do split).
    # Treina-se com o alvo escalado (SVR/MLP precisam de alvo ~O(1)); as
    # METRICAS, porem, sao calculadas no esforco BRUTO (regime do artigo), o
    # que se obtem invertendo a escala das predicoes. Guardamos o scaler.
    X = MinMaxScaler().fit_transform(X)
    y_scaler = MinMaxScaler().fit(y_raw.reshape(-1, 1))
    y = y_scaler.transform(y_raw.reshape(-1, 1)).ravel()

    # Passo 6 — holdout 70:30 (mesmos indices para escalado e bruto).
    idx = np.arange(len(y))
    itr, ite = train_test_split(idx, test_size=TEST_SIZE,
                                random_state=RANDOM_STATE)
    return dict(name=name, feat_cols=feat_cols, target=target,
                n_features=cfg["n_features"], drop=cfg["drop"],
                y_scaler=y_scaler,
                X_train=X[itr], X_test=X[ite],
                y_train=y[itr], y_test=y[ite],            # escalado (treino)
                y_train_raw=y_raw[itr], y_test_raw=y_raw[ite],  # bruto (metricas)
                check_ok=ok, check_got=got, check_ref=ref)
