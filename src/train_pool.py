"""
T2 — Geração do Pool de Modelos Base
====================================

Treina o pool de regressores INDIVIDUAIS (sem ensembles) sobre cada base e
gera as "Matrizes Gabarito": as predições de cada modelo no treino e no teste.
Essas matrizes alimentam a Seleção (SES-GA / DES) e a Combinação.

Por que o pool NÃO tem ensembles
--------------------------------
Random Forest, XGBoost, AdaBoost, Bagging, ExtraTrees etc. já SÃO ensembles.
Colocá-los como base de outro ensemble não faz sentido (alerta do professor).
O pool é montado por PARADIGMA de aprendizado, garantindo diversidade real:

    eager (modelo paramétrico ajustado no treino):
        SVR (kernel RBF), MLP (rede neural), Linear, Ridge, Lasso, ElasticNet
    lazy (baseado em instâncias):
        kNN
    árvore rasa:
        DecisionTree (profundidade limitada)

Os regressores-base vêm do scikit-learn; TODO o mecanismo de seleção,
combinação e validação estatística é de implementação própria (NumPy).

Diferença em relação ao código antigo
-------------------------------------
- Sem RF/XGB/CatBoost/Bagging/ExtraTrees/GaussianNB.
- Sem np.clip(pred, 0, 1): como o MinMax é ajustado só no treino, predições no
  teste podem legitimamente cair fora de [0,1]; clipar distorceria o erro.
- Lê o contrato de dados de src.dataset_loader (fonte única de verdade), em vez
  de CSVs de um loader paralelo.

API pública
-----------
    build_pool() -> dict[str, estimator]
    train_pool(X_train, y_train, X_test) -> dict (modelos + matrizes)
    run_train_pool(verbose) -> dict[name -> resultado]   # todas as bases
"""

from __future__ import annotations

import os
import json
import numpy as np

from sklearn.svm import SVR
from sklearn.model_selection import cross_val_predict, KFold
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression

from src.dataset_loader import RANDOM_STATE, OUT_DIR, load_all


# ─── Definição do pool (apenas modelos individuais) ──────────────────────────
def build_pool() -> dict:
    """Retorna {sigla: estimador} — paradigmas diversos, nenhum ensemble.

    5 modelos, um por paradigma de aprendizado (diversidade real):
        LR  — linear, eager, paramétrico
        SVR — margem/kernel RBF, eager, não-linear
        MLP — conexionista (rede neural), eager
        kNN — baseado em instâncias, lazy
        DT  — baseado em árvore (partição por regras)
    """
    return {
        "LR":  LinearRegression(),
        "SVR": SVR(kernel="rbf", C=10.0, gamma="scale"),
        "MLP": MLPRegressor(hidden_layer_sizes=(64,), activation="relu",
                            learning_rate_init=0.01, max_iter=800,
                            random_state=RANDOM_STATE),
        "kNN": KNeighborsRegressor(n_neighbors=5),
        "DT":  DecisionTreeRegressor(max_depth=4, random_state=RANDOM_STATE),
    }


# ─── Treinamento e geração das matrizes gabarito ─────────────────────────────
def train_pool(X_train: np.ndarray, y_train: np.ndarray,
               X_test: np.ndarray, dataset_name: str = "",
               verbose: bool = True) -> dict:
    """
    Treina cada modelo do pool e retorna as predições no treino e no teste.

    Returns
    -------
    dict com:
        model_names        : list[str]
        models             : dict[str, estimator] treinados
        pred_matrix_train  : (N_train, M) predições no treino
        pred_matrix_test   : (N_test,  M) predições no teste
    """
    pool = build_pool()
    names = list(pool.keys())
    M = len(names)
    pm_train = np.zeros((X_train.shape[0], M), dtype=np.float64)
    pm_train_oof = np.zeros((X_train.shape[0], M), dtype=np.float64)
    pm_test = np.zeros((X_test.shape[0], M), dtype=np.float64)
    trained = {}

    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    if verbose:
        print(f"[T2] {dataset_name:<11} treinando {M} modelos (+OOF 5-fold)...", end="")
    for i, (name, model) in enumerate(pool.items()):
        # Predições OUT-OF-FOLD para a SELEÇÃO (cada linha prevista por modelo
        # que NÃO a treinou) — evita o vazamento de selecionar no proprio treino.
        pm_train_oof[:, i] = cross_val_predict(model, X_train, y_train, cv=cv)
        # Modelo final treinado em todo o treino → predições de treino e teste
        model.fit(X_train, y_train)
        pm_train[:, i] = model.predict(X_train)   # SEM clip
        pm_test[:, i] = model.predict(X_test)
        trained[name] = model
    if verbose:
        print(" ok")

    return {
        "model_names": names,
        "models": trained,
        "pred_matrix_train": pm_train,
        "pred_matrix_train_oof": pm_train_oof,
        "pred_matrix_test": pm_test,
    }


# ─── Persistência ────────────────────────────────────────────────────────────
def save_matrices(name: str, result: dict):
    os.makedirs(OUT_DIR, exist_ok=True)
    p = os.path.join(OUT_DIR, name)
    np.save(f"{p}_pred_matrix_train.npy", result["pred_matrix_train"])
    np.save(f"{p}_pred_matrix_train_oof.npy", result["pred_matrix_train_oof"])
    np.save(f"{p}_pred_matrix_test.npy", result["pred_matrix_test"])


def save_registry(all_results: dict[str, dict]):
    os.makedirs(OUT_DIR, exist_ok=True)
    any_res = next(iter(all_results.values()))
    registry = {
        "contract_version": "2.0",
        "model_siglas": any_res["model_names"],
        "note": "modelos individuais (sem ensembles); predições NÃO clipadas",
        "matrix_format": {
            "shape": "(N_instances, M_models)",
            "dtype": "float64",
            "axis_0": "instâncias", "axis_1": "modelos",
        },
        "datasets": {
            name: {
                "n_models": len(r["model_names"]),
                "shape_train": list(r["pred_matrix_train"].shape),
                "shape_test": list(r["pred_matrix_test"].shape),
            } for name, r in all_results.items()
        },
    }
    with open(os.path.join(OUT_DIR, "pool_registry.json"), "w", encoding="utf-8") as f:
        json.dump(registry, f, indent=2, ensure_ascii=False)


# ─── Pipeline principal ──────────────────────────────────────────────────────
def run_train_pool(verbose: bool = True) -> dict[str, dict]:
    """Treina o pool em todas as bases disponíveis (via dataset_loader)."""
    print("=" * 64)
    print("  T2 — Pool de modelos base (individuais, sem ensembles)")
    print("=" * 64)
    splits = load_all(verbose=False)
    results = {}
    for name, s in splits.items():
        r = train_pool(s["X_train"], s["y_train"], s["X_test"], name, verbose)
        save_matrices(name, r)
        results[name] = r
    if results:
        save_registry(results)
    print("-" * 64)
    print(f"[T2] ✓ pool treinado em {len(results)} base(s) "
          f"({len(build_pool())} modelos) → matrizes em {OUT_DIR}/")
    return results


if __name__ == "__main__":
    run_train_pool()
