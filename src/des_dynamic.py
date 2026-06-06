"""
T5 - des_dynamic.py
Responsável: A
Implementa a Seleção Dinâmica de Ensemble (DES) usando k-NN local.
Lógica: para cada instância de teste, encontra os k vizinhos mais próximos
no conjunto de treino e seleciona o(s) modelo(s) com menor erro local.

AUTONOMIA: roda 100% com a Matriz Gabarito fictícia (mock) do passo T2.
Quando M entregar os dados reais, basta substituir os arrays.
"""

import numpy as np
from evaluator import evaluate_all


class DynamicEnsembleSelector:
    """
    Seleção Dinâmica de Ensemble baseada em k-NN local.

    Parâmetros
    ----------
    k : int
        Número de vizinhos para definir a região de competência.
    threshold : float
        Percentual dos melhores modelos a selecionar (ex: 0.3 = top 30%).
    """

    def __init__(self, k: int = 7, threshold: float = 0.3):
        self.k = k
        self.threshold = threshold
        # Serão preenchidos em fit()
        self.X_train = None          # Features de treino (N, F)
        self.gabarito = None         # Matriz de previsões (N, M) — cada coluna = 1 modelo
        self.y_train = None          # Alvo real de treino (N,)
        self.n_models = None

    # ------------------------------------------------------------------
    # 1. TREINO: memoriza X_train, gabarito e y_train
    # ------------------------------------------------------------------
    def fit(self, X_train: np.ndarray, gabarito: np.ndarray, y_train: np.ndarray):
        """
        Parâmetros
        ----------
        X_train  : array (N_treino, N_features)
        gabarito : array (N_treino, M_modelos)  ← contrato de interface com M
        y_train  : array (N_treino,)
        """
        self.X_train  = np.array(X_train, dtype=float)
        self.gabarito = np.array(gabarito, dtype=float)
        self.y_train  = np.array(y_train, dtype=float)
        self.n_models = gabarito.shape[1]
        print(f"[DES] fit() | {self.X_train.shape[0]} instâncias treino | {self.n_models} modelos no pool")
        return self

    # ------------------------------------------------------------------
    # 2. DISTÂNCIA EUCLIDIANA — encontra k vizinhos
    # ------------------------------------------------------------------
    def _get_neighbors(self, x_query: np.ndarray) -> np.ndarray:
        """Retorna índices dos k vizinhos mais próximos de x_query."""
        diffs = self.X_train - x_query          # broadcasting (N, F)
        dists = np.sqrt(np.sum(diffs ** 2, axis=1))
        return np.argsort(dists)[:self.k]       # índices dos k menores

    # ------------------------------------------------------------------
    # 3. COMPETÊNCIA LOCAL — erro de cada modelo na vizinhança
    # ------------------------------------------------------------------
    def _local_competence(self, neighbor_idx: np.ndarray) -> np.ndarray:
        """
        Calcula o MAE local de cada modelo nos k vizinhos.
        Retorna array (M_modelos,) com o erro local de cada um.
        """
        y_local    = self.y_train[neighbor_idx]              # (k,)
        preds_local = self.gabarito[neighbor_idx, :]         # (k, M)
        # MAE local por modelo
        local_errors = np.mean(np.abs(preds_local - y_local[:, np.newaxis]), axis=0)
        return local_errors  # (M,)

    # ------------------------------------------------------------------
    # 4. PREDIÇÃO: seleciona modelos competentes e faz média
    # ------------------------------------------------------------------
    def predict_single(self, x_query: np.ndarray, model_predictions: np.ndarray) -> float:
        """
        Prediz para UMA instância de teste.

        Parâmetros
        ----------
        x_query           : array (N_features,)
        model_predictions : array (M_modelos,) — predições de TODOS os modelos
                            para esta instância de teste.

        Retorna
        -------
        float : média das predições dos modelos selecionados
        """
        neighbor_idx   = self._get_neighbors(x_query)
        local_errors   = self._local_competence(neighbor_idx)

        # Seleciona top-threshold% modelos com MENOR erro local
        n_select = max(1, int(np.ceil(self.n_models * self.threshold)))
        selected = np.argsort(local_errors)[:n_select]

        return float(np.mean(model_predictions[selected]))

    def predict(self, X_test: np.ndarray, pool_predictions: np.ndarray) -> np.ndarray:
        """
        Prediz para TODO o conjunto de teste.

        Parâmetros
        ----------
        X_test           : array (N_teste, N_features)
        pool_predictions : array (N_teste, M_modelos)

        Retorna
        -------
        array (N_teste,)
        """
        X_test = np.array(X_test, dtype=float)
        pool_predictions = np.array(pool_predictions, dtype=float)
        preds = np.array([
            self.predict_single(X_test[i], pool_predictions[i])
            for i in range(len(X_test))
        ])
        print(f"[DES] predict() | {len(preds)} instâncias preditas | k={self.k} | top={self.threshold*100:.0f}% modelos")
        return preds


# -----------------------------------------------------------------
# Teste isolado com MOCK DATA — rode: python des_dynamic.py
# -----------------------------------------------------------------
if __name__ == "__main__":
    np.random.seed(42)

    # Simula dados fictícios (substitua pelos reais de M depois)
    N_TRAIN   = 283   # 70% de 405 (Finnish)
    N_TEST    = 122   # 30% de 405
    N_FEAT    = 31    # features selecionadas do Finnish
    M_MODELS  = 11    # 11 regressores do artigo

    print("\n[MOCK] Gerando dados fictícios para teste isolado do DES...")
    X_train_mock = np.random.rand(N_TRAIN, N_FEAT)
    y_train_mock = np.random.uniform(500, 60000, N_TRAIN)
    # Gabarito: previsões de cada modelo no treino (contrato de interface com M)
    gabarito_mock = np.column_stack([
        y_train_mock * np.random.uniform(0.8, 1.2, N_TRAIN)
        for _ in range(M_MODELS)
    ])

    X_test_mock = np.random.rand(N_TEST, N_FEAT)
    y_test_mock = np.random.uniform(500, 60000, N_TEST)
    pool_test_mock = np.column_stack([
        y_test_mock * np.random.uniform(0.8, 1.2, N_TEST)
        for _ in range(M_MODELS)
    ])

    # Instancia e testa o DES
    des = DynamicEnsembleSelector(k=7, threshold=0.3)
    des.fit(X_train_mock, gabarito_mock, y_train_mock)
    y_pred_des = des.predict(X_test_mock, pool_test_mock)

    # Avalia
    evaluate_all(y_test_mock, y_pred_des, model_name="DES (Mock)")
    print("\n[OK] des_dynamic.py funcionando corretamente.")
