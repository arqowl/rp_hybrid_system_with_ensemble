"""
Combinação de predições do ensemble
====================================

Etapa de COMBINAÇÃO do sistema híbrido. Por enquanto disponibiliza a
agregação por MÉDIA SIMPLES (baseline do artigo original). A combinação
PONDERADA POR COMPETÊNCIA LOCAL (contribuição da Etapa 1) será adicionada
aqui, mantendo a média como baseline para comparação.

Convenção de máscara
--------------------
`mask` é um vetor binário de tamanho M (modelos do pool). Modelos com 1
entram na combinação. Vazio → usa todos (degenera para média global).
"""

from __future__ import annotations
import numpy as np


def _active_idx(n_models: int, mask=None) -> np.ndarray:
    if mask is None:
        return np.arange(n_models)
    mask = np.asarray(mask).ravel()
    idx = np.where(mask == 1)[0]
    return idx if len(idx) else np.arange(n_models)


def combine_mean(pred_matrix: np.ndarray, mask=None) -> np.ndarray:
    """Média aritmética simples das predições dos modelos selecionados."""
    pred_matrix = np.asarray(pred_matrix, dtype=float)
    idx = _active_idx(pred_matrix.shape[1], mask)
    return np.mean(pred_matrix[:, idx], axis=1)


# Ponto de extensão da Etapa 1 (placeholder explícito):
# def combine_local_competence(...): peso ∝ 1/erro local na vizinhança k-NN.
