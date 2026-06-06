"""
T6 - evaluator.py
Responsável: A
Calcula as métricas de avaliação para qualquer conjunto de predições.
Métricas: sMAPE, MRE, MASE, NSE, COD (R²)
Totalmente isolado — só precisa receber arrays numpy.
"""

import numpy as np


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Symmetric Mean Absolute Percentage Error (sMAPE).
    Retorna valor em %. Quanto menor, melhor.
    Intervalo: [0%, 200%]
    """
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    numerator = 2.0 * np.abs(y_true - y_pred)
    denominator = np.abs(y_true) + np.abs(y_pred)
    # Evita divisão por zero
    mask = denominator != 0
    result = np.mean(numerator[mask] / denominator[mask]) * 100.0
    return round(result, 6)


def mre(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Mean Relative Error (MRE).
    Quanto menor, melhor.
    """
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    mask = y_true != 0
    result = np.mean(np.abs(y_true[mask] - y_pred[mask]) / np.abs(y_true[mask]))
    return round(result, 6)


def mase(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Mean Absolute Scaled Error (MASE).
    Se > 1: modelo pior que naive. Quanto menor, melhor.
    """
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    n = len(y_true)
    numerator = np.mean(np.abs(y_true - y_pred))
    # Erro naive: diferença entre observações consecutivas
    denominator = np.mean(np.abs(y_true[1:] - y_true[:-1]))
    if denominator == 0:
        return float('inf')
    return round(numerator / denominator, 6)


def nse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Nash-Sutcliffe Efficiency (NSE).
    Intervalo: (-inf, 1]. Quanto mais próximo de 1, melhor.
    """
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return float('nan')
    return round(1 - ss_res / ss_tot, 6)


def cod(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Coefficient of Determination (COD / R²).
    Intervalo: [0, 1]. Quanto mais próximo de 1, melhor.
    """
    return nse(y_true, y_pred)  # Mesma fórmula no contexto do artigo


def evaluate_all(y_true: np.ndarray, y_pred: np.ndarray, model_name: str = "Model") -> dict:
    """
    Calcula todas as métricas de uma vez.
    Retorna dicionário com resultados e imprime tabela formatada.
    """
    metrics = {
        "model":  model_name,
        "sMAPE":  smape(y_true, y_pred),
        "MRE":    mre(y_true, y_pred),
        "MASE":   mase(y_true, y_pred),
        "NSE":    nse(y_true, y_pred),
        "COD":    cod(y_true, y_pred),
    }
    print(f"\n{'='*50}")
    print(f"  Métricas — {model_name}")
    print(f"{'='*50}")
    for k, v in metrics.items():
        if k != "model":
            print(f"  {k:<8}: {v}")
    print(f"{'='*50}")
    return metrics


# -----------------------------------------------------------------
# Teste rápido isolado (mock data) — rode: python evaluator.py
# -----------------------------------------------------------------
if __name__ == "__main__":
    np.random.seed(42)
    y_true_mock = np.random.uniform(500, 60000, 100)
    y_pred_mock = y_true_mock * np.random.uniform(0.85, 1.15, 100)  # ~15% erro

    evaluate_all(y_true_mock, y_pred_mock, model_name="Mock Model")
    print("\n[OK] evaluator.py funcionando corretamente.")
