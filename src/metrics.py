"""
metrics.py — As 5 metricas de avaliacao do artigo (Eqs. 2 a 6).

Todas as metricas sao razoes/percentuais e portanto INVARIANTES A ESCALA
(nao mudam se y for normalizado ou nao). y = real, yhat = predito.
"""
import numpy as np


def smape(y, yhat):
    """Eq. 2 — symmetric MAPE em %, limites 0%..200%."""
    y = np.asarray(y, float); yhat = np.asarray(yhat, float)
    denom = np.abs(y) + np.abs(yhat)
    denom[denom == 0] = 1e-12          # evita divisao por zero
    return 100.0 * np.mean(2.0 * np.abs(y - yhat) / denom)


def mre(y, yhat):
    """Eq. 3 — Mean Relative Error."""
    y = np.asarray(y, float); yhat = np.asarray(yhat, float)
    yy = np.where(y == 0, 1e-12, y)
    return np.mean(np.abs(y - yhat) / np.abs(yy))


def mase(y, yhat):
    """Eq. 4 — Mean Absolute Scaled Error.

    Denominador = erro do forecast ingenuo (1-passo) sobre a serie de y,
    na ordem dada do conjunto. >1 indica desempenho ruim.
    """
    y = np.asarray(y, float); yhat = np.asarray(yhat, float)
    num = np.mean(np.abs(y - yhat))
    naive = np.mean(np.abs(np.diff(y)))
    naive = naive if naive != 0 else 1e-12
    return num / naive


def nse(y, yhat):
    """Eq. 5 — Nash-Sutcliffe Efficiency."""
    y = np.asarray(y, float); yhat = np.asarray(yhat, float)
    den = np.sum((y - np.mean(y)) ** 2)
    den = den if den != 0 else 1e-12
    return 1.0 - np.sum((y - yhat) ** 2) / den


def cod(y, yhat):
    """Eq. 6 — Coefficient of Determination.

    O artigo apresenta NSE e COD com a MESMA forma algebrica (Eqs. 5 e 6
    sao identicas). Replicamos fielmente: aqui o COD e reportado como o
    quadrado do coeficiente de correlacao de Pearson entre y e yhat, que e
    a interpretacao usual de 'coefficient of determination' e e o que
    produz valores em [0,1] coerentes com a Tabela 4/5 do artigo.
    Ver secao 'Pressupostos' do relatorio.
    """
    y = np.asarray(y, float); yhat = np.asarray(yhat, float)
    if np.std(y) == 0 or np.std(yhat) == 0:
        return 0.0
    r = np.corrcoef(y, yhat)[0, 1]
    return r ** 2


def all_metrics(y, yhat):
    """Retorna dict com as 5 metricas, na ordem das Tabelas 4 e 5."""
    return {
        "sMAPE": smape(y, yhat),
        "MRE": mre(y, yhat),
        "MASE": mase(y, yhat),
        "NSE": nse(y, yhat),
        "COD": cod(y, yhat),
    }
