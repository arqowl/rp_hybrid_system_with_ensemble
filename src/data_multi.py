"""
data_multi.py — Carregador generalizado para VARIAS bases de SEE, com dois
modos de pre-processamento que reaproveitam exatamente as fases de pipeline.py:

  - "baseline" : limpeza minima (remove ids/datas/strings/vazamento e
                 constantes), MinMax nas features, alvo MinMax (treina escalado,
                 metricas no bruto), split aleatorio. Analogo a v1.
  - "melhorado": baseline + REGRAS AUTOMATICAS de diagnostico, por base:
                 (a) remove variancia quase-nula; (b) remove uma de cada par de
                 features com |correlacao|>0.98; (c) remove features com
                 informacao mutua ~ 0 com o alvo; (d) log1p no alvo se |skew|>1;
                 (e) split estratificado por quantis do alvo. Analogo a v2.

Cada base tem um config minimo (alvo + colunas a sempre descartar: ids, datas,
strings e VAZAMENTO — colunas derivadas do esforco ou so conhecidas ao fim).
O resto e decidido automaticamente e REGISTRADO (para "identificar o
pre-processamento de cada base").

Bases incluidas (todas de estimacao de esforco de software; sem duplicatas):
  finnish, maxwell (artigo), desharnais, china, kitchenham, coc81.
Bases descartadas por NAO serem de esforco: debutanizer (sensor de processo),
abalone (idade de molusco).
"""
import os
import numpy as np
import pandas as pd
from scipy.io import arff
from scipy.stats import skew
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import mutual_info_regression

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "data", "raw")

TEST_SIZE = 0.30
NZV_THRESH = 1e-3      # variancia quase-nula (apos MinMax) -> descarta
CORR_THRESH = 0.98     # |correlacao| acima disso -> redundante, descarta uma
MI_THRESH = 1e-6       # informacao mutua <= isso -> sem relacao com o alvo
SKEW_THRESH = 1.0      # |skew| do alvo acima disso -> aplica log1p

# Escala ordinal dos fatores COCOMO (vl<l<n<h<vh<xh).
COCOMO_MAP = {"vl": 1, "l": 2, "n": 3, "h": 4, "vh": 5, "xh": 6}

# Config minimo por base. drop_always = ids/datas/strings/VAZAMENTO.
CONFIG = {
    "finnish": dict(fmt="csv", file="Finnish407.csv", target="Worksup",
                    drop_always=["Case_Number", "Project_tech_ID", "YK",
                                 "Project_name", "Business_names",
                                 "Protype_names", "Hardware_names", "Duration"]),
    "maxwell": dict(fmt="arff", file="maxwell.arff", target="Effort",
                    drop_always=["Duration", "Time"]),
    "desharnais": dict(fmt="csv", file="desharnais.csv", target="Effort",
                       drop_always=["id", "Project", "Length", "YearEnd"]),
    "china": dict(fmt="arff", file="china.arff", target="Effort",
                  drop_always=["ID", "N_effort", "PDR_AFP", "PDR_UFP",
                               "NPDR_AFP", "NPDU_UFP", "Duration"]),
    "kitchenham": dict(fmt="arff", file="kitchenham.arff", target="Actual.effort",
                       drop_always=["Project", "Actual.start.date",
                                    "Estimated.completion.date", "Actual.duration"]),
    "coc81": dict(fmt="arff", file="coc81-dem.arff", target="effort",
                  drop_always=["id", "defects", "months"]),
    # ---- regressao generica (NAO sao esforco de software) ----
    "debutanizer": dict(fmt="arff", file="phpWT77lf.arff", target="y",
                        drop_always=[]),
    "abalone": dict(fmt="arff", file="abalone.arff", target="rings",
                    drop_always=[]),
}

# Categoria de cada base (para o relatorio deixar claro o que e SEE e o que nao e)
CATEGORIA = {
    "finnish": "SEE", "maxwell": "SEE", "desharnais": "SEE",
    "china": "SEE", "kitchenham": "SEE", "coc81": "SEE",
    "debutanizer": "regressao generica (nao-SEE)",
    "abalone": "regressao generica (nao-SEE)",
}

# Procedencia (fonte) de cada base — usada no relatorio
FONTE = {
    "finnish": "TIEKE (Finlandia), 1997; repositorio PROMISE/figshare. Base do artigo.",
    "maxwell": "K.D. Maxwell, banco comercial finlandes; Applied Statistics for "
               "Software Managers (Prentice-Hall, 2002); Zenodo. Base do artigo.",
    "desharnais": "J.M. Desharnais (1989), tese de mestrado, UQAM; 81 projetos "
                  "canadenses; repositorio PROMISE.",
    "china": "China dataset (499 projetos por pontos de funcao); repositorio "
             "PROMISE / derivado do ISBSG.",
    "kitchenham": "Kitchenham, Pfleeger, McColl & Eagan (2002), 'An empirical study "
                  "of maintenance and development estimation accuracy'; PROMISE.",
    "coc81": "B. Boehm, Software Engineering Economics (Prentice-Hall, 1981); "
             "63 projetos COCOMO; repositorio PROMISE (coc81).",
    "debutanizer": "Coluna debutanizadora (sensor virtual de processo quimico). "
                   "Fortuna et al., Soft Sensors for Monitoring and Control of "
                   "Industrial Processes (Springer, 2007); benchmark do OpenML. "
                   "NAO e estimacao de esforco.",
    "abalone": "Abalone (prever idade/aneis a partir de medidas fisicas). "
               "Nash et al. (1994); UCI Machine Learning Repository. "
               "NAO e estimacao de esforco.",
}

DATASETS = ["finnish", "maxwell", "desharnais", "china", "kitchenham", "coc81",
            "debutanizer", "abalone"]


class LogMinMax:
    """log1p + MinMax no alvo; inverse_transform desfaz ambos -> esforco bruto.
    A reconstrucao em log e LIMITADA a faixa observada (+/- meia amplitude) antes
    do expm1, para evitar explosoes numericas quando um modelo extrapola muito
    acima do esforco visto no treino."""
    def fit(self, y):
        l = np.log1p(np.asarray(y, float)); self.mn, self.mx = float(l.min()), float(l.max()); return self
    def transform(self, y):
        l = np.log1p(np.asarray(y, float)); return (l - self.mn) / (self.mx - self.mn)
    def inverse_transform(self, a):
        a = np.asarray(a, float).ravel()
        l = a * (self.mx - self.mn) + self.mn
        pad = 0.5 * (self.mx - self.mn)
        l = np.clip(l, self.mn - pad, self.mx + pad)         # evita expm1 -> inf
        return np.expm1(l).reshape(-1, 1)


def _read_arff(path):
    """Leitor ARFF tolerante: aceita atributos string/date, linhas @class,
    comentarios, e dados separados por virgula OU por espaco. Devolve um
    DataFrame de strings (a conversao numerica/categorica e feita em _encode)."""
    attrs, data_lines, in_data = [], [], False
    with open(path, encoding="utf-8", errors="replace") as f:
        for raw in f:
            s = raw.strip().lstrip("\ufeff")
            if not s or s.startswith("%"):
                continue
            low = s.lower()
            if in_data:
                data_lines.append(s)
            elif low.startswith("@attribute") or low.startswith("@class"):
                # @attribute NOME TIPO...  (NOME pode estar entre aspas)
                parts = s.split(None, 2)
                if len(parts) >= 2:
                    attrs.append(parts[1].strip("'\""))
            elif low.startswith("@data"):
                in_data = True
            # @relation e outras linhas @ sao ignoradas
    if not data_lines:
        return pd.DataFrame(columns=attrs)
    sep_comma = any("," in ln for ln in data_lines[:5])
    rows = []
    for ln in data_lines:
        toks = [t.strip().strip("'\"") for t in (ln.split(",") if sep_comma else ln.split())]
        rows.append(toks)
    ncol = len(attrs)
    rows = [r[:ncol] + [None] * (ncol - len(r)) for r in rows]   # ajusta largura
    return pd.DataFrame(rows, columns=attrs)


def _load_raw(cfg):
    path = os.path.join(RAW, cfg["file"])
    if cfg["fmt"] == "csv":
        return pd.read_csv(path)
    return _read_arff(path)


def _encode(df):
    """Converte cada coluna em numerica. Detecta categorica de forma robusta
    (independe do dtype): se algum valor nao-nulo nao for numerico, trata como
    categorica (ordinal p/ niveis COCOMO; factorize p/ nominais). '?'/''/'nan'
    e o sentinela -1 viram NaN."""
    df = df.copy()
    NA = {"?": None, "": None, "nan": None, "None": None, "NaN": None}
    for c in df.columns:
        s = df[c].astype(str).str.strip().replace(NA)
        num = pd.to_numeric(s, errors="coerce")
        if num.notna().sum() == s.notna().sum():              # tudo numerico
            df[c] = num
        elif set(s.dropna().unique()) <= set(COCOMO_MAP):     # fator COCOMO (ordinal)
            df[c] = s.map(COCOMO_MAP)
        else:                                                  # nominal generico
            df[c] = pd.Series(pd.factorize(s)[0], index=s.index).astype(float)
    return df.replace(-1, np.nan)


def _make_target_scaler(y_raw, use_log):
    if use_log:
        return LogMinMax().fit(y_raw)
    return MinMaxScaler().fit(np.asarray(y_raw, float).reshape(-1, 1))


def build(name, mode="baseline", split_seed=42, cap=None):
    """Devolve o dict 'd' (compativel com pipeline.py) + um registro do
    pre-processamento aplicado (d['prep']). Se 'cap' for dado e a base tiver
    mais linhas que isso, subamostra (usado so na robustez para bases grandes)."""
    cfg = CONFIG[name]
    target = cfg["target"]
    df = _load_raw(cfg)

    # remove colunas sempre descartadas (ids/datas/strings/vazamento)
    drop_ids = [c for c in cfg["drop_always"] if c in df.columns]
    df = df.drop(columns=drop_ids)
    df = _encode(df)

    # ausentes: remove linhas invalidas (Finnish: esforco 0) e NaN
    if name == "finnish":
        df = df[df[target] > 0]
    n_antes = len(df)
    df = df.dropna().reset_index(drop=True)
    n_removidas = n_antes - len(df)

    # subamostra opcional (so robustez de bases grandes)
    if cap and len(df) > cap:
        df = df.sample(n=cap, random_state=split_seed).reset_index(drop=True)

    feat_cols = [c for c in df.columns if c != target]
    X = df[feat_cols].astype(float).copy()
    y_raw = df[target].astype(float).values

    prep = {"drop_ids": drop_ids, "n_linhas": len(df), "n_removidas_na": int(n_removidas),
            "drop_nzv": [], "drop_corr": [], "drop_mi": [], "log_alvo": False}

    # --- constantes (variancia quase-nula): removidas nos DOIS modos, pois
    #     constantes nao informam nada e quebram correlacao/escala ---
    Xmm0 = pd.DataFrame(MinMaxScaler().fit_transform(X), columns=feat_cols)
    const = Xmm0.columns[Xmm0.var() <= 1e-12].tolist()
    if const:
        X = X.drop(columns=const); feat_cols = [c for c in feat_cols if c not in const]
        prep["drop_const"] = sorted(const)
    else:
        prep["drop_const"] = []

    if mode == "melhorado":
        # (a) variancia quase-nula
        Xmm = pd.DataFrame(MinMaxScaler().fit_transform(X), columns=feat_cols)
        nzv = Xmm.columns[Xmm.var() < NZV_THRESH].tolist()
        if nzv and len(nzv) < len(feat_cols):
            X = X.drop(columns=nzv); feat_cols = [c for c in feat_cols if c not in nzv]
            prep["drop_nzv"] = sorted(nzv)
        # (b) redundancia |corr|>0.98 (remove a 2a de cada par)
        corr = X.corr().abs()
        drop_corr = set()
        for i in range(len(feat_cols)):
            for j in range(i + 1, len(feat_cols)):
                if corr.iloc[i, j] > CORR_THRESH and feat_cols[j] not in drop_corr:
                    drop_corr.add(feat_cols[j])
        if drop_corr and len(drop_corr) < len(feat_cols):
            X = X.drop(columns=list(drop_corr)); feat_cols = [c for c in feat_cols if c not in drop_corr]
            prep["drop_corr"] = sorted(drop_corr)
        # (c) informacao mutua ~ 0 com o alvo
        Xmm = MinMaxScaler().fit_transform(X)
        mi = mutual_info_regression(Xmm, y_raw, random_state=42)
        mi_drop = [feat_cols[k] for k in range(len(feat_cols)) if mi[k] <= MI_THRESH]
        if mi_drop and len(mi_drop) < len(feat_cols):
            keep = [c for c in feat_cols if c not in mi_drop]
            X = X[keep]; feat_cols = keep
            prep["drop_mi"] = sorted(mi_drop)
        # (d) log1p no alvo se assimetrico
        prep["log_alvo"] = bool(abs(skew(y_raw)) > SKEW_THRESH)

    # escala das features (MinMax) e do alvo (treina escalado; metrica no bruto)
    Xs = MinMaxScaler().fit_transform(X.values)
    yscl = _make_target_scaler(y_raw, prep["log_alvo"])
    y = yscl.transform(np.asarray(y_raw, float).reshape(-1, 1)).ravel() if not prep["log_alvo"] \
        else yscl.transform(y_raw)

    # split (estratificado por quantis se 'melhorado', senao aleatorio)
    idx = np.arange(len(y))
    if mode == "melhorado":
        q = min(10, max(3, len(y) // 8))
        bins = pd.qcut(y_raw, q=q, labels=False, duplicates="drop")
        itr, ite = train_test_split(idx, test_size=TEST_SIZE, random_state=split_seed, stratify=bins)
    else:
        itr, ite = train_test_split(idx, test_size=TEST_SIZE, random_state=split_seed)

    prep["n_features"] = len(feat_cols)
    prep["features"] = feat_cols
    return dict(name=name, mode=mode, feat_cols=feat_cols, target=target,
                n_features=len(feat_cols), y_scaler=yscl, prep=prep,
                X_train=Xs[itr], X_test=Xs[ite], y_train=y[itr], y_test=y[ite],
                y_train_raw=y_raw[itr], y_test_raw=y_raw[ite],
                check_ok=True, check_got=None, check_ref=None)
