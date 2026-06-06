"""
T7 - statistical_tests.py
Responsável: A
Implementa a validação estatística com:
  1. Teste de Friedman (não-paramétrico, compara N modelos em K datasets)
  2. Post-hoc de Bonferroni-Dunn (compara cada modelo vs. o proposto OES)
Referência: Demšar (2006) — padrão em comparações de ML.

AUTONOMIA: roda com tabela de desempenho fictícia.
Quando M entregar os resultados reais, substitua a tabela mock.
"""

import numpy as np
from scipy import stats
from scipy.stats import friedmanchisquare
import warnings
warnings.filterwarnings("ignore")

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_OK = True
except ImportError:
    MATPLOTLIB_OK = False


# ------------------------------------------------------------------
# 1. TESTE DE FRIEDMAN
# ------------------------------------------------------------------
def friedman_test(results_matrix: np.ndarray, model_names: list) -> dict:
    """
    Teste de Friedman sobre uma matriz de resultados.

    Parâmetros
    ----------
    results_matrix : array (K_datasets, N_modelos)
        Cada linha = 1 dataset, cada coluna = 1 modelo.
        Valores: métrica de erro (ex: sMAPE) — menor = melhor.
    model_names : list de str

    Retorna
    -------
    dict com estatística, p-value e ranks médios
    """
    K, N = results_matrix.shape
    assert len(model_names) == N, "Número de nomes != número de colunas"

    # Calcula ranks por linha (dataset) — rank 1 = melhor (menor erro)
    ranks = np.zeros_like(results_matrix, dtype=float)
    for i in range(K):
        row = results_matrix[i]
        # argsort duplo: rank 1 para o menor
        order = np.argsort(row)
        rank_row = np.empty_like(order, dtype=float)
        rank_row[order] = np.arange(1, N + 1)
        ranks[i] = rank_row

    avg_ranks = np.mean(ranks, axis=0)

    # Estatística de Friedman: F = 12K / [N(N+1)] * [sum(R²) - N(N+1)²/4]
    sum_sq_ranks = np.sum(avg_ranks ** 2)
    F_friedman = (12 * K / (N * (N + 1))) * (sum_sq_ranks - N * (N + 1) ** 2 / 4)

    # p-value via chi-quadrado com (N-1) graus de liberdade
    p_value = 1 - stats.chi2.cdf(F_friedman, df=N - 1)

    print("\n" + "="*55)
    print("  TESTE DE FRIEDMAN")
    print("="*55)
    print(f"  Estatística de Friedman : {F_friedman:.4f}")
    print(f"  p-value                 : {p_value:.6f}")
    print(f"  Significativo (p<0.05)  : {'✅ SIM' if p_value < 0.05 else '❌ NÃO'}")
    print("\n  Ranks médios por modelo:")
    for name, rank in sorted(zip(model_names, avg_ranks), key=lambda x: x[1]):
        bar = "█" * int(rank * 3)
        print(f"    {name:<20} rank={rank:.3f}  {bar}")
    print("="*55)

    return {
        "F_statistic": F_friedman,
        "p_value": p_value,
        "avg_ranks": dict(zip(model_names, avg_ranks)),
        "rank_matrix": ranks,
        "significant": p_value < 0.05
    }


# ------------------------------------------------------------------
# 2. POST-HOC: BONFERRONI-DUNN
# ------------------------------------------------------------------
def bonferroni_dunn_test(friedman_result: dict, control_model: str,
                         n_datasets: int, alpha: float = 0.05) -> dict:
    """
    Compara cada modelo contra o modelo de controle (OES proposto).
    Usa a Diferença Crítica (CD) de Bonferroni-Dunn.

    Parâmetros
    ----------
    friedman_result : dict retornado por friedman_test()
    control_model   : nome do modelo proposto (ex: 'OES')
    n_datasets      : número de datasets (K)
    alpha           : nível de significância

    Retorna
    -------
    dict com diferenças de rank e se cada comparação é significativa
    """
    avg_ranks  = friedman_result["avg_ranks"]
    N          = len(avg_ranks)
    control_rank = avg_ranks[control_model]

    # Erro padrão de Bonferroni-Dunn
    se = np.sqrt(N * (N + 1) / (6 * n_datasets))

    # z crítico para Bonferroni (corrigido pelo número de comparações)
    n_comparisons = N - 1
    z_alpha = stats.norm.ppf(1 - alpha / (2 * n_comparisons))
    cd = z_alpha * se  # Diferença Crítica

    print("\n" + "="*55)
    print(f"  POST-HOC: BONFERRONI-DUNN  (controle: {control_model})")
    print("="*55)
    print(f"  Diferença Crítica (CD) : {cd:.4f}")
    print(f"  z crítico (Bonferroni) : {z_alpha:.4f}")
    print(f"  alpha ajustado         : {alpha/n_comparisons:.5f}\n")

    results = {}
    for model, rank in avg_ranks.items():
        if model == control_model:
            continue
        diff = abs(rank - control_rank)
        significant = diff > cd
        symbol = "✅ MELHOR" if (rank > control_rank and significant) else (
                 "❌ PIOR"  if (rank < control_rank and significant) else "— empate")
        print(f"  {control_model} vs {model:<18}  |diff|={diff:.3f}  CD={cd:.3f}  {symbol}")
        results[model] = {"rank_diff": diff, "significant": significant,
                          "cd": cd, "control_better": rank > control_rank}

    print("="*55)
    return results


# ------------------------------------------------------------------
# 3. PLOT: Critical Difference Diagram (estilo Demšar)
# ------------------------------------------------------------------
def plot_cd_diagram(friedman_result: dict, control_model: str,
                    n_datasets: int, alpha: float = 0.05,
                    save_path: str = None):
    """Gera o diagrama de Diferença Crítica no estilo Demšar (2006)."""
    if not MATPLOTLIB_OK:
        print("[AVISO] matplotlib não disponível. Pulando plot.")
        return

    avg_ranks = friedman_result["avg_ranks"]
    N = len(avg_ranks)
    se = np.sqrt(N * (N + 1) / (6 * n_datasets))
    n_comp = N - 1
    z_alpha = stats.norm.ppf(1 - alpha / (2 * n_comp))
    cd = z_alpha * se

    sorted_models = sorted(avg_ranks.items(), key=lambda x: x[1])
    names  = [m[0] for m in sorted_models]
    ranks  = [m[1] for m in sorted_models]

    fig, ax = plt.subplots(figsize=(10, 3))
    ax.set_xlim(1, N)
    ax.set_ylim(-1, 1)
    ax.axhline(0, color='black', lw=1)

    for i, (name, rank) in enumerate(zip(names, ranks)):
        color = 'red' if name == control_model else 'steelblue'
        ax.plot(rank, 0, 'o', color=color, ms=8, zorder=5)
        va = 'bottom' if i % 2 == 0 else 'top'
        offset = 0.15 if i % 2 == 0 else -0.15
        ax.text(rank, offset, f"{name}\n({rank:.2f})",
                ha='center', va=va, fontsize=8, color=color)

    # Barra de CD a partir do modelo de controle
    ctrl_rank = avg_ranks[control_model]
    ax.annotate('', xy=(ctrl_rank + cd, 0.5), xytext=(ctrl_rank, 0.5),
                arrowprops=dict(arrowstyle='<->', color='green', lw=2))
    ax.text(ctrl_rank + cd / 2, 0.6, f'CD={cd:.2f}', ha='center',
            fontsize=8, color='green')

    ax.set_title(f'Critical Difference Diagram (Demšar) — controle: {control_model}',
                 fontsize=11, fontweight='bold')
    ax.set_xlabel('Rank médio (menor = melhor)')
    ax.set_yticks([])
    plt.tight_layout()

    path = save_path or "outputs/cd_diagram.png"
    plt.savefig(f"/home/claude/projeto_oes/{path}", dpi=150, bbox_inches='tight')
    print(f"\n[PLOT] Diagrama salvo em: {path}")
    plt.close()


# ------------------------------------------------------------------
# Teste isolado com MOCK DATA — rode: python statistical_tests.py
# ------------------------------------------------------------------
if __name__ == "__main__":
    np.random.seed(42)

    # Mock: 2 datasets (Finnish, Maxwell) x 15 modelos
    # Valores de sMAPE (menor = melhor) — baseados no artigo
    model_names = ["SVM","RF","MLP","kNN","DT","ET","LR",
                   "ADA","CAT","XGB","NB","BG","Static","DES","OES"]

    # Simula resultados próximos aos do artigo (Finnish / Maxwell)
    results_mock = np.array([
        # Finnish (linha 1) — sMAPE
        [48.1, 29.1, 38.7, 35.6, 37.5, 27.7, 42.9,
         37.6, 33.5, 30.1, 38.7, 43.4, 27.8, 33.8, 23.9],
        # Maxwell (linha 2) — sMAPE
        [37.3, 18.2, 45.0, 35.0, 30.1, 18.5, 65.2,
         24.8, 36.4, 18.6, 39.1, 20.5, 23.5, 16.5, 15.0],
    ])

    # 1. Friedman
    fr_result = friedman_test(results_mock, model_names)

    # 2. Bonferroni-Dunn (OES como controle)
    if fr_result["significant"]:
        bd_result = bonferroni_dunn_test(fr_result, control_model="OES",
                                         n_datasets=2, alpha=0.05)
    else:
        print("\n[INFO] Friedman não significativo — post-hoc não aplicável.")

    # 3. Plot CD Diagram
    plot_cd_diagram(fr_result, control_model="OES", n_datasets=2, alpha=0.05,
                    save_path="outputs/cd_diagram.png")

    print("\n[OK] statistical_tests.py funcionando corretamente.")
