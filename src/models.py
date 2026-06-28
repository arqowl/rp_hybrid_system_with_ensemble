"""
models.py — Pool de regressores (Tabela 3 do artigo).

Parametros replicados EXATAMENTE conforme a coluna "Parameter Description".

Implementacoes proprias (sem dependencia externa):
  - XGBoost  -> implementado na mao (classe XGBoostRegressor abaixo): gradient
                boosting de arvores para erro quadratico, com peso de folha
                REGULARIZADO  w = -G/(H+lambda)  e ganho de split pela objetivo
                regularizado (a matematica do XGBoost). Parametros do artigo:
                n_estimators=50, max_depth=3, eta=0.1.
  - CatBoost -> mantido como GradientBoostingRegressor do scikit-learn com os
                MESMOS hiperparametros do artigo (n_estimators=10,
                learning_rate=1, depth=2). O CatBoost "de verdade" depende de
                ordered boosting + oblivious trees, cuja reimplementacao fiel
                seria complexa e fora do escopo de "codigo simples"; documentado.
  - DES      -> tambem e implementacao propria (ver ensembles.py), nao usa lib.

Escolhas documentadas onde o artigo lista variantes:
  - SVM: artigo lista kernels linear/rbf/polynomial (degree=3). Reportamos o
         kernel 'rbf' como linha "SVM" (default mais comum). [Pressupostos]
  - kNN: artigo lista k=3/5/7. Reportamos k=5 (valor central). [Pressupostos]
  - MLP: 1 camada oculta, learning_rate=0.01; nro de neuronios nao especificado
         -> default simples (100). [Pressupostos]
  - Naive Bayes (regressao): NB e nativamente classificador. Adaptacao minima:
         discretiza o alvo em quantis e prediz a media do bin (wrapper abaixo).
"""
import numpy as np
from sklearn.svm import SVR
from sklearn.ensemble import (RandomForestRegressor, ExtraTreesRegressor,
                              AdaBoostRegressor, BaggingRegressor,
                              GradientBoostingRegressor)
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.base import BaseEstimator, RegressorMixin

RS = 42


class NaiveBayesRegressor(BaseEstimator, RegressorMixin):
    """Adaptacao minima de NB para alvo continuo (documentada).

    Discretiza y em `bins` quantis, treina GaussianNB como classificador do
    bin e prediz a media do alvo no bin previsto.
    """
    def __init__(self, bins=10):
        self.bins = bins

    def fit(self, X, y):
        y = np.asarray(y, float)
        self.clf = GaussianNB()
        # quantis (descarta duplicados se houver empates)
        edges = np.unique(np.quantile(y, np.linspace(0, 1, self.bins + 1)))
        self.edges_ = edges
        lab = np.clip(np.digitize(y, edges[1:-1]), 0, len(edges) - 2)
        self.bin_mean_ = np.array(
            [y[lab == b].mean() if np.any(lab == b) else y.mean()
             for b in range(len(edges) - 1)])
        self.clf.fit(X, lab)
        return self

    def predict(self, X):
        lab = self.clf.predict(X)
        return self.bin_mean_[np.clip(lab, 0, len(self.bin_mean_) - 1)]


# ---------------------------------------------------------------------------
# XGBoost implementado na mao (erro quadratico)
# ---------------------------------------------------------------------------
# Para a perda L = 1/2 (y - pred)^2:  gradiente g = pred - y ;  hessiana h = 1.
# Cada arvore minimiza a objetivo regularizada do XGBoost:
#   peso da folha            w* = -G / (H + lambda)
#   ganho de um split        0.5*[ G_L^2/(H_L+λ) + G_R^2/(H_R+λ)
#                                   - (G_L+G_R)^2/(H_L+H_R+λ) ] - gamma
# onde G,H sao as somas de g,h nas amostras do no. O update usa shrinkage (eta).
def _build_tree(X, g, h, depth, max_depth, lam, gamma, mcw):
    """Constroi recursivamente uma arvore XGBoost (exact greedy, vetorizado)."""
    G, H = float(g.sum()), float(h.sum())
    leaf_w = -G / (H + lam)
    if depth >= max_depth or len(g) < 2:
        return {"leaf": leaf_w}

    best_gain, best = 0.0, None
    base = G * G / (H + lam)
    for f in range(X.shape[1]):
        order = np.argsort(X[:, f], kind="mergesort")
        xs = X[order, f]; gs = g[order]; hs = h[order]
        Gl = np.cumsum(gs)[:-1]; Hl = np.cumsum(hs)[:-1]
        Gr = G - Gl;             Hr = H - Hl
        valid = (xs[:-1] != xs[1:]) & (Hl >= mcw) & (Hr >= mcw)
        if not valid.any():
            continue
        gain = 0.5 * (Gl**2 / (Hl + lam) + Gr**2 / (Hr + lam) - base) - gamma
        gain = np.where(valid, gain, -np.inf)
        k = int(np.argmax(gain))
        if gain[k] > best_gain:
            best_gain = float(gain[k])
            best = (f, (xs[k] + xs[k + 1]) / 2.0)

    if best is None:
        return {"leaf": leaf_w}
    f, thr = best
    m = X[:, f] <= thr
    return {"f": f, "thr": thr,
            "left": _build_tree(X[m], g[m], h[m], depth + 1, max_depth, lam, gamma, mcw),
            "right": _build_tree(X[~m], g[~m], h[~m], depth + 1, max_depth, lam, gamma, mcw)}


def _tree_predict(node, X):
    out = np.empty(len(X))
    for i, x in enumerate(X):
        nd = node
        while "leaf" not in nd:
            nd = nd["left"] if x[nd["f"]] <= nd["thr"] else nd["right"]
        out[i] = nd["leaf"]
    return out


class XGBoostRegressor(BaseEstimator, RegressorMixin):
    """XGBoost de regressao (erro quadratico) escrito do zero. Compativel com
    a API do scikit-learn (clone/cross_val_predict)."""

    def __init__(self, n_estimators=50, max_depth=3, eta=0.1,
                 reg_lambda=1.0, gamma=0.0, min_child_weight=1.0):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.eta = eta
        self.reg_lambda = reg_lambda
        self.gamma = gamma
        self.min_child_weight = min_child_weight

    def fit(self, X, y):
        X = np.asarray(X, float); y = np.asarray(y, float)
        self.base_ = float(np.mean(y))          # base_score = media (regressao)
        pred = np.full(len(y), self.base_)
        self.trees_ = []
        for _ in range(self.n_estimators):
            g = pred - y                         # gradiente do erro quadratico
            hh = np.ones_like(y)                 # hessiana = 1
            tree = _build_tree(X, g, hh, 0, self.max_depth,
                               self.reg_lambda, self.gamma, self.min_child_weight)
            pred = pred + self.eta * _tree_predict(tree, X)
            self.trees_.append(tree)
        return self

    def predict(self, X):
        X = np.asarray(X, float)
        pred = np.full(len(X), self.base_)
        for tree in self.trees_:
            pred = pred + self.eta * _tree_predict(tree, X)
        return pred


def build_pool():
    """Retorna dict {sigla: estimador} na ordem das Tabelas 4 e 5."""
    return {
        "SVM": SVR(kernel="rbf", degree=3),                       # rbf reportado
        "RF":  RandomForestRegressor(n_estimators=10, min_samples_leaf=1,
                                     random_state=RS),
        "MLP": MLPRegressor(hidden_layer_sizes=(100,), learning_rate_init=0.01,
                            max_iter=1000, random_state=RS),
        "kNN": KNeighborsRegressor(n_neighbors=5),                # k=5 reportado
        "DT":  DecisionTreeRegressor(criterion="squared_error", max_depth=2,
                                     random_state=RS),
        "ET":  ExtraTreesRegressor(n_estimators=100, min_samples_leaf=1,
                                   random_state=RS),
        "LR":  LinearRegression(),
        "ADA": AdaBoostRegressor(n_estimators=10, random_state=RS),
        "CAT": GradientBoostingRegressor(n_estimators=10, learning_rate=1.0,
                                         max_depth=2, random_state=RS),  # ~CatBoost
        "XGB": XGBoostRegressor(n_estimators=50, max_depth=3, eta=0.1),  # implementado na mao
        "NB":  NaiveBayesRegressor(bins=10),
        "BG":  BaggingRegressor(random_state=RS),
    }


POOL_ORDER = ["SVM", "RF", "MLP", "kNN", "DT", "ET", "LR",
              "ADA", "CAT", "XGB", "NB", "BG"]
