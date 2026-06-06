"""
T4 - Algoritmo Genético Multi-Objetivo (SES-GA Multi)
Responsável: M | Apoio: A

Descrição:
    Extensão direta de T3. Modifica o GA para otimizar DOIS objetivos
    simultaneamente:
        1. Maximizar R² (precisão do ensemble)
        2. Minimizar nº de regressores ativos (parcimônia)

    Implementa seleção de Pareto (NSGA-II simplificado) em NumPy puro.
    Retorna a Frente de Pareto com os melhores trade-offs.

    Dependência: reutiliza diretamente as funções de T3
        (crossover, mutação, inicialização, fitness R²).

Entregáveis:
    results/finnish_ses_ga_multi.json
    results/maxwell_ses_ga_multi.json
    results/finnish_pareto_front.npy
    results/maxwell_pareto_front.npy
    Módulo importável: ses_ga_multi.run_ses_ga_multi(...)
"""

import os
import json
import time
import numpy as np

# Reutiliza funções de T3 (parcimônia — evita duplicação de código)
from src.ses_ga_single import (
    pearson_r_squared,
    r_squared,
    ensemble_predict,
    initialize_population,
    single_point_crossover,
    bit_flip_mutation,
    tournament_selection,
    RANDOM_STATE,
)

from src.dataset_loader import RESULTS_DIR   # ancorado na raiz do projeto


# ─── Dominância de Pareto ─────────────────────────────────────────────────────
def dominates(obj1: np.ndarray, obj2: np.ndarray) -> bool:
    """
    Verifica se obj1 domina obj2 (maximização de ambos os objetivos).
    obj1 domina obj2 se: obj1 >= obj2 em todos os objetivos
                         E obj1 > obj2 em pelo menos um.

    Objetivos: [R², -n_models]  (ambos a maximizar)
    """
    return (np.all(obj1 >= obj2) and np.any(obj1 > obj2))


def fast_non_dominated_sort(objectives: np.ndarray) -> list:
    """
    Ordena a população em frentes de Pareto (NSGA-II).
    objectives: shape (pop_size, 2) — [R², -n_active]

    Retorna lista de listas de índices por frente (frente 0 = Pareto ótima).
    """
    pop_size = len(objectives)
    domination_count = np.zeros(pop_size, dtype=int)
    dominated_by = [[] for _ in range(pop_size)]
    fronts = [[]]

    for i in range(pop_size):
        for j in range(pop_size):
            if i == j:
                continue
            if dominates(objectives[i], objectives[j]):
                dominated_by[i].append(j)
            elif dominates(objectives[j], objectives[i]):
                domination_count[i] += 1

        if domination_count[i] == 0:
            fronts[0].append(i)

    front_idx = 0
    while front_idx < len(fronts) and fronts[front_idx]:
        next_front = []
        for i in fronts[front_idx]:
            for j in dominated_by[i]:
                domination_count[j] -= 1
                if domination_count[j] == 0:
                    next_front.append(j)
        front_idx += 1
        if next_front:
            fronts.append(next_front)

    return fronts


def crowding_distance(objectives: np.ndarray, front: list) -> np.ndarray:
    """
    Calcula a distância de aglomeração (crowding distance) para um front.
    Garante diversidade dentro da frente de Pareto.

    Returns:
        np.ndarray de shape (len(front),) com distâncias
    """
    n = len(front)
    distances = np.zeros(n)

    if n <= 2:
        distances[:] = np.inf
        return distances

    front_obj = objectives[front]  # shape (n, 2)

    for m in range(front_obj.shape[1]):
        order = np.argsort(front_obj[:, m])
        distances[order[0]] = np.inf
        distances[order[-1]] = np.inf

        obj_range = front_obj[order[-1], m] - front_obj[order[0], m]
        if obj_range == 0:
            continue
        for k in range(1, n - 1):
            distances[order[k]] += (
                (front_obj[order[k + 1], m] - front_obj[order[k - 1], m])
                / obj_range
            )
    return distances


# ─── Avaliação multi-objetivo ─────────────────────────────────────────────────
def evaluate_multi(pop: np.ndarray, pred_matrix: np.ndarray,
                   y_true: np.ndarray) -> np.ndarray:
    """
    Avalia cada cromossomo em dois objetivos:
        obj[0] = R² (maximizar)
        obj[1] = -n_active / n_models  (maximizar → minimiza modelos)

    Returns:
        np.ndarray de shape (pop_size, 2)
    """
    n_models = pop.shape[1]
    objectives = np.zeros((len(pop), 2))

    for i, chrom in enumerate(pop):
        y_pred = ensemble_predict(pred_matrix, chrom)
        r2 = pearson_r_squared(y_true, y_pred)
        n_active = max(int(chrom.sum()), 1)
        # Normaliza número de modelos: -n_active/n_models ∈ [-1, 0]
        parsimony = -n_active / n_models

        objectives[i, 0] = r2
        objectives[i, 1] = parsimony

    return objectives


# ─── Seleção por ranking de Pareto + crowding distance ───────────────────────
def pareto_selection(pop: np.ndarray, objectives: np.ndarray,
                     n_select: int) -> np.ndarray:
    """
    Seleciona n_select indivíduos ordenados por:
        1. Frente de Pareto (rank menor = melhor)
        2. Crowding distance (maior = melhor)
    """
    fronts = fast_non_dominated_sort(objectives)
    selected_idx = []

    for front in fronts:
        if len(selected_idx) + len(front) <= n_select:
            selected_idx.extend(front)
        else:
            remaining = n_select - len(selected_idx)
            cd = crowding_distance(objectives, front)
            sorted_by_cd = sorted(zip(cd, front), reverse=True)
            selected_idx.extend([idx for _, idx in sorted_by_cd[:remaining]])
            break

    return pop[selected_idx[:n_select]]


# ─── GA Multi-objetivo Principal ─────────────────────────────────────────────
def run_ses_ga_multi(pred_matrix_train: np.ndarray,
                     y_train: np.ndarray,
                     pred_matrix_test: np.ndarray = None,
                     y_test: np.ndarray = None,
                     pop_size: int = 60,
                     n_generations: int = 100,
                     crossover_rate: float = 0.8,
                     mutation_rate: float = 0.02,
                     tournament_size: int = 3,
                     elitism: int = 2,
                     random_state: int = RANDOM_STATE,
                     verbose: bool = True) -> dict:
    """
    Executa o SES-GA Multi-objetivo (R² × parcimônia).

    Returns:
        dict com:
            pareto_front: lista de cromossomos na frente de Pareto final
            pareto_objectives: objetivos [R², -n_active/M] de cada solução
            best_balanced: cromossomo com melhor soma ponderada (50/50)
            best_r2: cromossomo com maior R² na frente
            best_parsimonious: cromossomo com menos modelos na frente
            history_r2: melhor R² por geração
            history_pareto_size: tamanho da frente de Pareto por geração
            runtime_s: tempo de execução
    """
    rng = np.random.default_rng(random_state)
    n_models = pred_matrix_train.shape[1]
    t_start = time.time()

    # Inicialização
    pop = initialize_population(pop_size, n_models, rng)
    objectives = evaluate_multi(pop, pred_matrix_train, y_train)

    history_r2 = []
    history_pareto_size = []

    if verbose:
        print(f"\n  [SES-GA-Multi] Iniciando: pop={pop_size}, ger={n_generations}, "
              f"M={n_models} modelos")
        print(f"  {'Ger':>5} | {'Melhor R²':>10} | {'|Pareto|':>9} | "
              f"{'Min #Mod':>9}")
        print("  " + "-" * 44)

    for gen in range(n_generations):
        # Ordena por Pareto
        fronts = fast_non_dominated_sort(objectives)
        pareto_front_idx = fronts[0]

        best_r2_gen = float(np.max(objectives[:, 0]))
        pareto_size = len(pareto_front_idx)
        # Menor número de modelos na frente de Pareto
        min_models_gen = int(
            np.min([-objectives[i, 1] * n_models for i in pareto_front_idx])
        )

        history_r2.append(best_r2_gen)
        history_pareto_size.append(pareto_size)

        if verbose and (gen % 10 == 0 or gen == n_generations - 1):
            print(f"  {gen+1:>5} | {best_r2_gen:>10.6f} | {pareto_size:>9} | "
                  f"{min_models_gen:>9}")

        # Elitismo: preserva soluções da frente de Pareto
        n_elite = min(elitism, len(pareto_front_idx))
        elite = pop[pareto_front_idx[:n_elite]].copy()

        # Seleção por Pareto + crowding
        n_offspring = pop_size - n_elite
        selected = pareto_selection(pop, objectives, n_select=n_offspring)

        # Crossover + Mutação
        offspring = []
        for i in range(0, len(selected) - 1, 2):
            c1, c2 = single_point_crossover(selected[i], selected[i + 1],
                                            crossover_rate, rng)
            offspring.append(bit_flip_mutation(c1, mutation_rate, rng))
            offspring.append(bit_flip_mutation(c2, mutation_rate, rng))
        if len(offspring) < n_offspring:
            offspring.append(bit_flip_mutation(selected[-1], mutation_rate, rng))

        offspring = np.array(offspring[:n_offspring])
        pop = np.vstack([elite, offspring])
        objectives = evaluate_multi(pop, pred_matrix_train, y_train)

    # Extrai frente de Pareto final
    final_fronts = fast_non_dominated_sort(objectives)
    pareto_idx = final_fronts[0]
    pareto_chromosomes = pop[pareto_idx]
    pareto_obj = objectives[pareto_idx]

    # Solução mais balanceada: maior soma(R² + (-parsimônia normalizada para [0,1]))
    # Normaliza ambos para [0,1] antes de somar
    r2_vals = pareto_obj[:, 0]
    pars_vals = -pareto_obj[:, 1]  # n_active/n_models ∈ (0,1]

    r2_norm = (r2_vals - r2_vals.min()) / (np.ptp(r2_vals) + 1e-12)
    pars_norm = 1.0 - (pars_vals - pars_vals.min()) / (np.ptp(pars_vals) + 1e-12)

    balanced_score = 0.5 * r2_norm + 0.5 * pars_norm
    best_balanced_local = int(np.argmax(balanced_score))
    best_r2_local = int(np.argmax(r2_vals))
    best_pars_local = int(np.argmin(pars_vals))

    # Avaliação no teste
    def eval_test(chrom):
        if pred_matrix_test is None or y_test is None:
            return None
        return r_squared(y_test, ensemble_predict(pred_matrix_test, chrom))

    runtime = time.time() - t_start

    if verbose:
        print(f"\n  [SES-GA-Multi] ✓ Concluído em {runtime:.2f}s")
        print(f"  Frente de Pareto final: {len(pareto_idx)} soluções")
        best_chrom = pareto_chromosomes[best_balanced_local]
        print(f"  Solução balanceada → R²={pareto_obj[best_balanced_local,0]:.4f}, "
              f"#modelos={int(best_chrom.sum())}")
        best_r2_chrom = pareto_chromosomes[best_r2_local]
        print(f"  Maior R²           → R²={pareto_obj[best_r2_local,0]:.4f}, "
              f"#modelos={int(best_r2_chrom.sum())}")
        best_p_chrom = pareto_chromosomes[best_pars_local]
        print(f"  Mais parcimonioso  → R²={pareto_obj[best_pars_local,0]:.4f}, "
              f"#modelos={int(best_p_chrom.sum())}")

    return {
        "pareto_front": pareto_chromosomes.tolist(),
        "pareto_objectives": pareto_obj.tolist(),   # [R², -n_active/M]
        "n_pareto_solutions": len(pareto_idx),
        "best_balanced": {
            "chromosome": pareto_chromosomes[best_balanced_local].tolist(),
            "r2_train": float(pareto_obj[best_balanced_local, 0]),
            "n_models": int(pareto_chromosomes[best_balanced_local].sum()),
            "r2_test": eval_test(pareto_chromosomes[best_balanced_local]),
        },
        "best_r2": {
            "chromosome": pareto_chromosomes[best_r2_local].tolist(),
            "r2_train": float(pareto_obj[best_r2_local, 0]),
            "n_models": int(pareto_chromosomes[best_r2_local].sum()),
            "r2_test": eval_test(pareto_chromosomes[best_r2_local]),
        },
        "best_parsimonious": {
            "chromosome": pareto_chromosomes[best_pars_local].tolist(),
            "r2_train": float(pareto_obj[best_pars_local, 0]),
            "n_models": int(pareto_chromosomes[best_pars_local].sum()),
            "r2_test": eval_test(pareto_chromosomes[best_pars_local]),
        },
        "history_r2": history_r2,
        "history_pareto_size": history_pareto_size,
        "runtime_s": runtime,
        "hyperparams": {
            "pop_size": pop_size,
            "n_generations": n_generations,
            "crossover_rate": crossover_rate,
            "mutation_rate": mutation_rate,
            "tournament_size": tournament_size,
            "elitism": elitism,
            "random_state": random_state,
        }
    }

# ─── Pipeline principal ──────────────────────────────────────────────────────
def _to_jsonable(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def run_ses_ga_multi_pipeline(verbose: bool = True) -> dict:
    """Roda o SES-GA multi-objetivo (R² × parcimônia) em todas as bases (T2)."""
    from src.dataset_loader import datasets_with_pool, load_pool_arrays, load_oof_train

    print("=" * 60)
    print("  T4 — SES-GA multi-objetivo (R² × parcimônia) — NSGA-II, seleção OOF")
    print("=" * 60)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    results = {}
    for dname in datasets_with_pool():
        _, pred_test, y_train, y_test = load_pool_arrays(dname)
        pred_oof = load_oof_train(dname)            # fitness em OOF (sem vazamento)
        print(f"\n── {dname.upper()} ──")
        res = run_ses_ga_multi(pred_oof, y_train, pred_test, y_test, verbose=verbose)
        results[dname] = res
        out = os.path.join(RESULTS_DIR, f"{dname}_ses_ga_multi.json")
        with open(out, "w", encoding="utf-8") as f:
            json.dump(json.loads(json.dumps(res, default=_to_jsonable)), f, indent=2)
        np.save(os.path.join(RESULTS_DIR, f"{dname}_pareto_front.npy"),
                np.array(res["pareto_front"]))
    print(f"\n[T4] ✓ SES-GA-multi em {len(results)} base(s) → {RESULTS_DIR}/")
    return results


if __name__ == "__main__":
    run_ses_ga_multi_pipeline()
