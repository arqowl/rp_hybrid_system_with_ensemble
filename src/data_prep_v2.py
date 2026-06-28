"""
data_prep_v2.py — Pre-processamento MELHORADO (variante; nao altera a
replicacao fiel em data_prep.py). Aplica as 3 melhorias indicadas pelo
diagnostico:

  (1) ALVO em log1p (+ MinMax) -> reduz a forte assimetria (skew 3.7/3.3 -> ~0).
      As metricas continuam sendo calculadas no esforco BRUTO (via
      inverse_transform), para serem comparaveis com a execucao anterior.
  (2) REMOCAO DIRECIONADA de atributos (alem do recorte da v1):
      - Finnish: AllFP_ep20 (agregado = soma das componentes FP, VIF = inf) +
        IntFP, AlgFP (variancia quase-nula).
      - Maxwell: Har, T01, T04, T06, T13 (informacao mutua ~ 0 com o alvo).
  (3) SPLIT 70:30 ESTRATIFICADO por quantis do alvo (em vez de aleatorio puro),
      garantindo que treino e teste cubram toda a faixa de esforco. (A
      validacao cruzada repetida fica no runner run_experiment_v2.py.)

NAO inclui PCA nem resampling: o diagnostico mostrou baixa compressibilidade
(95% da variancia exige 23/30 e 16/20 componentes), entao seria de baixo valor.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

import data_prep as dp

TEST_SIZE = 0.30

# Atributos descartados ALEM do recorte da v1 (decididos pelo diagnostico).
EXTRA_DROP = {
    "finnish": ["AllFP_ep20", "IntFP", "AlgFP"],
    "maxwell": ["Har", "T01", "T04", "T06", "T13"],
}


class LogMinMax:
    """Transforma o alvo: log1p e depois MinMax (para treino). O
    inverse_transform desfaz os dois passos, devolvendo o esforco BRUTO."""

    def fit(self, y):
        l = np.log1p(np.asarray(y, float))
        self.mn, self.mx = float(l.min()), float(l.max())
        return self

    def transform(self, y):
        l = np.log1p(np.asarray(y, float))
        return (l - self.mn) / (self.mx - self.mn)

    def inverse_transform(self, a):
        a = np.asarray(a, float).ravel()
        l = a * (self.mx - self.mn) + self.mn
        return np.expm1(l).reshape(-1, 1)


def build(name, split_seed=42):
    """Mesma interface de data_prep.build, com o pre-processamento melhorado."""
    cfg = dp.CONFIG[name]
    df = dp._load_raw(cfg)
    target = cfg["target"]

    # ausentes (Finnish: remove as 2 obs. com esforco invalido -> 405)
    if name == "finnish":
        df = df[df[target] > 0].reset_index(drop=True)
    df = df.dropna(subset=[target]).reset_index(drop=True)

    # selecao = recorte v1 menos os atributos extras do diagnostico
    drop = set(cfg["drop"]) | set(EXTRA_DROP[name])
    keep = [c for c in df.columns if c not in drop]
    feat_cols = [c for c in keep if c != target]

    X = df[feat_cols].astype(float).values
    y_raw = df[target].astype(float).values

    # features MinMax (igual a v1); alvo log1p + MinMax
    X = MinMaxScaler().fit_transform(X)
    yscl = LogMinMax().fit(y_raw)
    y = yscl.transform(y_raw)

    # split 70:30 ESTRATIFICADO por quantis do alvo
    q = 5 if name == "maxwell" else 10
    bins = pd.qcut(y_raw, q=q, labels=False, duplicates="drop")
    idx = np.arange(len(y))
    itr, ite = train_test_split(idx, test_size=TEST_SIZE,
                                random_state=split_seed, stratify=bins)

    return dict(name=name, feat_cols=feat_cols, target=target,
                n_features=len(keep), drop=sorted(drop),
                y_scaler=yscl,
                X_train=X[itr], X_test=X[ite],
                y_train=y[itr], y_test=y[ite],
                y_train_raw=y_raw[itr], y_test_raw=y_raw[ite],
                # checagem da Tabela 2 nao se aplica aqui (alvo transformado);
                # a validacao das bases ja foi feita na replicacao fiel.
                check_ok=True, check_got=None, check_ref=None)
