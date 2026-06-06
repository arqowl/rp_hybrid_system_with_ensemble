"""
T2 - Geração do Pool de Modelos Base
Responsável: M | Apoio: A

Descrição:
    Treina todos os regressores do pool (Tabela 3 do artigo) sobre o
    conjunto de treino e gera a Matriz Gabarito de treino.

    Matriz Gabarito: array NumPy de shape (N_train, M_models)
        - Cada coluna = predições de um modelo sobre o treino
        - Usada pela Seleção Dinâmica (T5) para medir competência local

Entregáveis:
    models/finnish_pool/   ← modelos .joblib
    models/maxwell_pool/   ← modelos .joblib
    data/finnish_pred_matrix_train.npy
    data/maxwell_pred_matrix_train.npy
    data/finnish_pred_matrix_test.npy
    data/maxwell_pred_matrix_test.npy
    data/pool_registry.json ← contrato de interface para T5

Autonomia de A (T5):
    Enquanto T2 não estiver pronto, A usa matrix_mock() para
    criar uma Matriz Gabarito fictícia com a mesma estrutura.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.svm import SVR
from sklearn.ensemble import (
    RandomForestRegressor, AdaBoostRegressor,
    BaggingRegressor, ExtraTreesRegressor,
    GradientBoostingRegressor
)
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

# ─── Configuração ────────────────────────────────────────────────────────────
RANDOM_STATE = 42
DATA_DIR = "data"
MODELS_DIR = "models"

# ─── Definição do Pool (Tabela 3 do artigo) ──────────────────────────────────
def build_pool() -> dict:
    """
    Retorna dicionário {sigla: instância do regressor} conforme Tabela 3.
    Parâmetros exatamente como descritos no artigo.
    """
    pool = {
        "SVM": SVR(kernel="rbf"),                               # kernel=rbf
        "RF":  RandomForestRegressor(n_estimators=10,
                                     min_samples_leaf=1,
                                     random_state=RANDOM_STATE),
        "MLP": MLPRegressor(hidden_layer_sizes=(100,),
                            learning_rate_init=0.01,
                            max_iter=500,
                            random_state=RANDOM_STATE),
        "kNN": KNeighborsRegressor(n_neighbors=5),              # k=3/5/7 → 5 default
        "DT":  DecisionTreeRegressor(criterion="squared_error",
                                     max_depth=2,
                                     random_state=RANDOM_STATE),
        "ET":  ExtraTreesRegressor(n_estimators=100,
                                   min_samples_leaf=1,
                                   random_state=RANDOM_STATE),
        "LR":  LinearRegression(),
        "ADA": AdaBoostRegressor(n_estimators=10,
                                 random_state=RANDOM_STATE),
        "CAT": CatBoostRegressor(iterations=10,
                                 learning_rate=1,
                                 depth=2,
                                 verbose=0,
                                 random_state=RANDOM_STATE),
        "XGB": XGBRegressor(n_estimators=50,
                            max_depth=3,
                            eta=0.1,
                            verbosity=0,
                            random_state=RANDOM_STATE),
        "NB":  GaussianNB(),                                     # Naive Bayes
        "BG":  BaggingRegressor(n_estimators=10,
                                random_state=RANDOM_STATE),
    }
    return pool


# ─── Mock para autonomia de A ────────────────────────────────────────────────
def matrix_mock(n_instances: int, m_models: int = 12,
                random_state: int = RANDOM_STATE) -> np.ndarray:
    """
    Gera uma Matriz Gabarito fictícia com a mesma estrutura da real.
    Usada por A (T5) para desenvolver o DES de forma independente.

    Args:
        n_instances: número de instâncias de treino (N_train)
        m_models: número de modelos no pool (padrão=12)
        random_state: semente para reprodutibilidade

    Returns:
        np.ndarray de shape (n_instances, m_models), dtype=float64
        valores em [0, 1] simulando predições normalizadas
    """
    rng = np.random.default_rng(random_state)
    # Simula predições com correlação moderada (mais realista que ruído puro)
    base = rng.uniform(0, 1, size=n_instances)
    noise = rng.normal(0, 0.15, size=(n_instances, m_models))
    matrix = np.clip(base[:, None] + noise, 0, 1)
    return matrix.astype(np.float64)


# ─── Treinamento do Pool ──────────────────────────────────────────────────────
def train_pool(X_train: np.ndarray, y_train: np.ndarray,
               X_test: np.ndarray, dataset_name: str) -> dict:
    """
    Treina cada modelo do pool e retorna:
        - modelos treinados
        - Matriz Gabarito de treino: shape (N_train, M_models)
        - Matriz de predições de teste: shape (N_test, M_models)
    """
    pool = build_pool()
    model_names = list(pool.keys())
    M = len(model_names)
    N_train = X_train.shape[0]
    N_test = X_test.shape[0]

    pred_matrix_train = np.zeros((N_train, M), dtype=np.float64)
    pred_matrix_test = np.zeros((N_test, M), dtype=np.float64)
    trained_models = {}

    print(f"\n[T2] Treinando pool para {dataset_name} ({M} modelos)...")

    # Cria diretório de modelos
    model_dir = os.path.join(MODELS_DIR, f"{dataset_name}_pool")
    os.makedirs(model_dir, exist_ok=True)

    for i, (name, model) in enumerate(pool.items()):
        try:
            model.fit(X_train, y_train)

            pred_train = model.predict(X_train)
            pred_test = model.predict(X_test)

            # Clipa para [0,1] — dados já normalizados pelo MinMax
            pred_matrix_train[:, i] = np.clip(pred_train, 0, 1)
            pred_matrix_test[:, i] = np.clip(pred_test, 0, 1)

            trained_models[name] = model

            # Salva modelo como .joblib
            model_path = os.path.join(model_dir, f"{name}.joblib")
            joblib.dump(model, model_path)

            print(f"  [{i+1:02d}/{M}] {name:<5} ✓  (salvo em {model_path})")

        except Exception as e:
            print(f"  [{i+1:02d}/{M}] {name:<5} ✗  ERRO: {e}")
            # Preenche coluna com mock se modelo falhar
            pred_matrix_train[:, i] = matrix_mock(N_train, 1,
                                                   RANDOM_STATE + i).ravel()
            pred_matrix_test[:, i] = matrix_mock(N_test, 1,
                                                  RANDOM_STATE + i + 100).ravel()

    return {
        "models": trained_models,
        "model_names": model_names,
        "pred_matrix_train": pred_matrix_train,
        "pred_matrix_test": pred_matrix_test,
        "model_dir": model_dir,
    }


# ─── Salvar matrizes e registro ───────────────────────────────────────────────
def save_artifacts(result: dict, dataset_name: str,
                   y_train: np.ndarray, y_test: np.ndarray):
    """Salva as matrizes NumPy e o registro do pool."""
    prefix = os.path.join(DATA_DIR, dataset_name)

    train_path = f"{prefix}_pred_matrix_train.npy"
    test_path = f"{prefix}_pred_matrix_test.npy"
    y_train_path = f"{prefix}_y_train.npy"
    y_test_path = f"{prefix}_y_test.npy"

    np.save(train_path, result["pred_matrix_train"])
    np.save(test_path, result["pred_matrix_test"])
    np.save(y_train_path, y_train)
    np.save(y_test_path, y_test)

    shape_train = result["pred_matrix_train"].shape
    shape_test = result["pred_matrix_test"].shape
    print(f"  → Matriz Gabarito treino:  {train_path}  shape={shape_train}")
    print(f"  → Matriz Gabarito teste:   {test_path}   shape={shape_test}")

    return {
        "train_matrix": train_path,
        "test_matrix": test_path,
        "y_train": y_train_path,
        "y_test": y_test_path,
        "shape_train": list(shape_train),
        "shape_test": list(shape_test),
    }


def save_pool_registry(finnish_info: dict, maxwell_info: dict,
                       finnish_result: dict, maxwell_result: dict):
    """
    Gera pool_registry.json — contrato de interface para T5 (DES).
    Define exatamente onde estão as matrizes e os modelos.
    """
    registry = {
        "contract_version": "1.0",
        "description": "Contrato T2 → T5. Pool de modelos treinados e Matrizes Gabarito.",
        "model_siglas": finnish_result["model_names"],
        "mock_function": "from train_pool import matrix_mock",
        "mock_usage": "matrix_mock(n_instances, m_models=12) → np.ndarray (N, M)",
        "datasets": {
            "finnish": finnish_info,
            "maxwell": maxwell_info,
        },
        "matrix_format": {
            "shape": "(N_instances, M_models)",
            "dtype": "float64",
            "values": "predições MinMax-normalizadas [0, 1]",
            "axis_0": "instâncias",
            "axis_1": "modelos na ordem: " + str(finnish_result["model_names"]),
        }
    }
    path = os.path.join(DATA_DIR, "pool_registry.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(registry, f, indent=2, ensure_ascii=False)
    print(f"\n[T2] Contrato de interface salvo: {path}")


# ─── Pipeline principal ──────────────────────────────────────────────────────
def run_train_pool():
    """
    Executa T2 completo.
    Depende dos CSVs gerados por T1 (dataset_loader.py).
    """
    print("=" * 60)
    print("  T2 — Geração do Pool de Modelos Base")
    print("=" * 60)

    os.makedirs(MODELS_DIR, exist_ok=True)

    # Carrega dados produzidos por T1
    datasets = {
        "finnish": {
            "train": os.path.join(DATA_DIR, "finnish_train.csv"),
            "test":  os.path.join(DATA_DIR, "finnish_test.csv"),
            "target": "Worksup",
        },
        "maxwell": {
            "train": os.path.join(DATA_DIR, "maxwell_train.csv"),
            "test":  os.path.join(DATA_DIR, "maxwell_test.csv"),
            "target": "Effort",
        },
    }

    registry_data = {}
    for dname, paths in datasets.items():
        train_df = pd.read_csv(paths["train"])
        test_df  = pd.read_csv(paths["test"])

        target = paths["target"]
        feature_cols = [c for c in train_df.columns if c != target]

        X_train = train_df[feature_cols].values
        y_train = train_df[target].values
        X_test  = test_df[feature_cols].values
        y_test  = test_df[target].values

        result = train_pool(X_train, y_train, X_test, dname)
        info = save_artifacts(result, dname, y_train, y_test)
        info["model_dir"] = result["model_dir"]
        info["n_models"] = len(result["model_names"])
        registry_data[dname] = (info, result)

    save_pool_registry(
        registry_data["finnish"][0], registry_data["maxwell"][0],
        registry_data["finnish"][1], registry_data["maxwell"][1],
    )

    print("\n[T2] ✓ Pool de modelos gerado com sucesso!")
    print(f"      Matrizes Gabarito em: ./{DATA_DIR}/")
    print(f"      Modelos .joblib em:   ./{MODELS_DIR}/")


if __name__ == "__main__":
    # Garante que T1 foi executado antes
    if not os.path.exists(os.path.join(DATA_DIR, "finnish_train.csv")):
        print("[T2] Dados de T1 não encontrados. Executando T1 primeiro...")
        import sys
        sys.path.insert(0, ".")
        from dataset_loader import run_pipeline
        run_pipeline()

    run_train_pool()
