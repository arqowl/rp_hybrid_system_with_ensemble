"""
T3 - Algoritmo Genético para Seleção Estática (SES-GA) — Padrão
Responsável: M | Apoio: A

Descrição:
    Implementa o GA em Python/NumPy puro focado apenas em R²
    (Coeficiente de Determinação / COD).
    Busca o subconjunto de modelos do pool que maximiza R²
    sobre o conjunto de treino — Static Ensemble Selection.

    Fluxo conforme artigo:
        1. Inicialização: população de cromossomos binários
        2. Avaliação: fitness = R² do ensemble selecionado
        3. Seleção: torneio
        4. Crossover: ponto único
        5. Mutação: bit flip
        6. Critério de parada: n_generations ou estagnação

Entregáveis:
    results/finnish_ses_ga.json
    results/maxwell_ses_ga.json
    Módulo importável: ses_ga_single.run_ses_ga(...)

Autonomia:
    Módulo 100% isolado. Usa apenas NumPy + Matrizes Gabarito de T2.
"""

import os
import json
import time
import numpy as np

# ─── Configuração ────────────────────────────────────────────────────────────
RANDOM_STATE = 42
RESULTS_DIR = "results"


# ─── Função de Fitness: R² / COD ─────────────────────────────────────────────
def r_squared(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Coeficiente de Determinação (R² / COD).
    Equação 6 do artigo:
        COD = 1 - Σ(yi - ŷi)² / Σ(yi - ȳ)²

    Retorna -inf se denominador for zero (target constante).
    """
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return -np.inf
    return float(1.0 - ss_res / ss_tot)


def pearson_r_squared(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Quadrado do coeficiente de Pearson (equação 1 do artigo).
    Fitness function do GA conforme Seção IV-D.
    """
    n = len(y_true)
    if n == 0:
        return -np.inf
    sum_a = np.sum(y_true)
    sum_p = np.sum(y_pred)
    sum_ap = np.sum(y_true * y_pred)
    sum_a2 = np.sum(y_true ** 2)
    sum_p2 = np.sum(y_pred ** 2)

    num = n * sum_ap - sum_a * sum_p
    den = np.sqrt(
        max(n * sum_a2 - sum_a ** 2, 1e-12) *
        max(n * sum_p2 - sum_p ** 2, 1e-12)
    )
    if den == 0:
        return 0.0
    r = num / den
    return float(r ** 2)


# ─── Ensemble: média das predições dos modelos selecionados ──────────────────
def ensemble_predict(pred_matrix: np.ndarray, chromosome: np.ndarray) -> np.ndarray:
    """
    Calcula a predição do ensemble (média) dos modelos ativos.

    Args:
        pred_matrix: shape (N, M) — predições de cada modelo
        chromosome: array binário de tamanho M

    Returns:
        array de shape (N,) com a predição média
    """
    active = np.where(chromosome == 1)[0]
    if len(active) == 0:
        # Cromossomo vazio → retorna média global (pior caso)
        return np.mean(pred_matrix, axis=1)
    return np.mean(pred_matrix[:, active], axis=1)


# ─── Inicialização da população ───────────────────────────────────────────────
def initialize_population(pop_size: int, n_models: int,
                           rng: np.random.Generator) -> np.ndarray:
    """
    Gera população inicial de cromossomos binários aleatórios.
    Garante que nenhum cromossomo seja completamente vazio.

    Returns:
        np.ndarray de shape (pop_size, n_models), dtype=int
    """
    pop = rng.integers(0, 2, size=(pop_size, n_models)).astype(int)
    # Força pelo menos 1 gene ativo por cromossomo
    for i in range(pop_size):
        if pop[i].sum() == 0:
            pop[i, rng.integers(0, n_models)] = 1
    return pop


# ─── Avaliação de fitness ─────────────────────────────────────────────────────
def evaluate_population(pop: np.ndarray, pred_matrix: np.ndarray,
                         y_true: np.ndarray) -> np.ndarray:
    """
    Calcula o fitness (Pearson R²) de cada cromossomo.

    Returns:
        np.ndarray de shape (pop_size,) com valores de fitness
    """
    fitness = np.zeros(len(pop))
    for i, chrom in enumerate(pop):
        y_pred = ensemble_predict(pred_matrix, chrom)
        fitness[i] = pearson_r_squared(y_true, y_pred)
    return fitness


# ─── Seleção por torneio ──────────────────────────────────────────────────────
def tournament_selection(pop: np.ndarray, fitness: np.ndarray,
                          n_select: int, tournament_size: int,
                          rng: np.random.Generator) -> np.ndarray:
    """
    Seleciona n_select indivíduos por torneio.
    """
    selected = np.zeros((n_select, pop.shape[1]), dtype=int)
    pop_size = len(pop)
    for i in range(n_select):
        candidates = rng.integers(0, pop_size, size=tournament_size)
        best = candidates[np.argmax(fitness[candidates])]
        selected[i] = pop[best]
    return selected


# ─── Crossover de ponto único ─────────────────────────────────────────────────
def single_point_crossover(parent1: np.ndarray, parent2: np.ndarray,
                            crossover_rate: float,
                            rng: np.random.Generator) -> tuple:
    """
    Crossover de ponto único com taxa crossover_rate.
    """
    if rng.random() < crossover_rate:
        point = rng.integers(1, len(parent1))
        child1 = np.concatenate([parent1[:point], parent2[point:]])
        child2 = np.concatenate([parent2[:point], parent1[point:]])
        return child1, child2
    return parent1.copy(), parent2.copy()


# ─── Mutação por bit flip ─────────────────────────────────────────────────────
def bit_flip_mutation(chromosome: np.ndarray, mutation_rate: float,
                       rng: np.random.Generator) -> np.ndarray:
    """
    Inverte cada bit com probabilidade mutation_rate.
    Garante pelo menos 1 gene ativo após mutação.
    """
    mutant = chromosome.copy()
    mask = rng.random(size=len(mutant)) < mutation_rate
    mutant[mask] = 1 - mutant[mask]
    if mutant.sum() == 0:
        mutant[rng.integers(0, len(mutant))] = 1
    return mutant


# ─── Algoritmo Genético Principal ────────────────────────────────────────────
def run_ses_ga(pred_matrix_train: np.ndarray,
               y_train: np.ndarray,
               pred_matrix_test: np.ndarray = None,
               y_test: np.ndarray = None,
               pop_size: int = 50,
               n_generations: int = 100,
               crossover_rate: float = 0.8,
               mutation_rate: float = 0.02,
               tournament_size: int = 3,
               elitism: int = 2,
               random_state: int = RANDOM_STATE,
               verbose: bool = True) -> dict:
    """
    Executa o SES-GA mono-objetivo (maximiza R²).

    Args:
        pred_matrix_train: shape (N_train, M_models) — Matriz Gabarito de treino
        y_train: array de targets de treino, shape (N_train,)
        pred_matrix_test: shape (N_test, M_models) — opcional para avaliação
        y_test: targets de teste — opcional
        pop_size: tamanho da população
        n_generations: número de gerações
        crossover_rate: taxa de crossover
        mutation_rate: taxa de mutação por bit
        tournament_size: tamanho do torneio de seleção
        elitism: quantos melhores indivíduos preservar por geração
        random_state: semente para reprodutibilidade
        verbose: imprime progresso

    Returns:
        dict com:
            best_chromosome: array binário (M,) com os modelos selecionados
            best_fitness: R² do melhor cromossomo (treino)
            best_r2_test: R² no teste (se fornecido)
            selected_models_idx: índices dos modelos selecionados
            n_models_selected: número de modelos no ensemble final
            history: lista de (geração, melhor_fitness, media_fitness)
            runtime_s: tempo de execução em segundos
    """
    rng = np.random.default_rng(random_state)
    n_models = pred_matrix_train.shape[1]
    t_start = time.time()

    # Inicialização
    pop = initialize_population(pop_size, n_models, rng)
    fitness = evaluate_population(pop, pred_matrix_train, y_train)

    best_idx = np.argmax(fitness)
    best_chromosome = pop[best_idx].copy()
    best_fitness = float(fitness[best_idx])
    history = []
    stagnation = 0

    if verbose:
        print(f"\n  [SES-GA] Iniciando: pop={pop_size}, ger={n_generations}, "
              f"M={n_models} modelos")
        print(f"  {'Ger':>5} | {'Melhor R²':>10} | {'Média R²':>10} | "
              f"{'#Modelos':>8}")
        print("  " + "-" * 42)

    for gen in range(n_generations):
        # Elitismo: preserva os `elitism` melhores
        elite_idx = np.argsort(fitness)[-elitism:]
        elite = pop[elite_idx].copy()

        # Seleção
        selected = tournament_selection(pop, fitness,
                                        n_select=pop_size - elitism,
                                        tournament_size=tournament_size,
                                        rng=rng)

        # Crossover + Mutação
        offspring = []
        for i in range(0, len(selected) - 1, 2):
            c1, c2 = single_point_crossover(selected[i], selected[i + 1],
                                            crossover_rate, rng)
            offspring.append(bit_flip_mutation(c1, mutation_rate, rng))
            offspring.append(bit_flip_mutation(c2, mutation_rate, rng))
        if len(offspring) < len(selected):
            offspring.append(bit_flip_mutation(selected[-1], mutation_rate, rng))

        offspring = np.array(offspring[:len(selected)])

        # Nova geração
        pop = np.vstack([elite, offspring])
        fitness = evaluate_population(pop, pred_matrix_train, y_train)

        gen_best_idx = np.argmax(fitness)
        gen_best = float(fitness[gen_best_idx])
        gen_mean = float(np.mean(fitness))

        if gen_best > best_fitness:
            best_fitness = gen_best
            best_chromosome = pop[gen_best_idx].copy()
            stagnation = 0
        else:
            stagnation += 1

        n_active = int(pop[gen_best_idx].sum())
        history.append((gen + 1, gen_best, gen_mean))

        if verbose and (gen % 10 == 0 or gen == n_generations - 1):
            print(f"  {gen+1:>5} | {gen_best:>10.6f} | {gen_mean:>10.6f} | "
                  f"{n_active:>8}")

        # Critério de parada antecipada (estagnação de 20 gerações)
        if stagnation >= 20:
            if verbose:
                print(f"  → Parada antecipada por estagnação na geração {gen+1}")
            break

    runtime = time.time() - t_start
    selected_idx = list(np.where(best_chromosome == 1)[0])

    # Avaliação no teste (se fornecido)
    best_r2_test = None
    if pred_matrix_test is not None and y_test is not None:
        y_pred_test = ensemble_predict(pred_matrix_test, best_chromosome)
        best_r2_test = r_squared(y_test, y_pred_test)

    if verbose:
        print(f"\n  [SES-GA] ✓ Concluído em {runtime:.2f}s")
        print(f"  Melhor R² treino: {best_fitness:.6f}")
        if best_r2_test is not None:
            print(f"  R² teste:         {best_r2_test:.6f}")
        print(f"  Modelos selecionados ({len(selected_idx)}): {selected_idx}")

    return {
        "best_chromosome": best_chromosome.tolist(),
        "best_fitness_train": best_fitness,
        "best_r2_test": best_r2_test,
        "selected_models_idx": selected_idx,
        "n_models_selected": len(selected_idx),
        "history": history,
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
def run_ses_ga_pipeline():
    import pandas as pd
    from train_pool import matrix_mock

    print("=" * 60)
    print("  T3 — Algoritmo Genético SES-GA (Mono-objetivo: R²)")
    print("=" * 60)

    os.makedirs(RESULTS_DIR, exist_ok=True)

    datasets = ["finnish", "maxwell"]
    all_results = {}

    for dname in datasets:
        print(f"\n{'─'*50}")
        print(f"  Dataset: {dname.upper()}")
        print(f"{'─'*50}")

        # Tenta carregar matrizes reais de T2; usa mock se não existirem
        train_path = os.path.join("data", f"{dname}_pred_matrix_train.npy")
        test_path  = os.path.join("data", f"{dname}_pred_matrix_test.npy")
        y_train_path = os.path.join("data", f"{dname}_y_train.npy")
        y_test_path  = os.path.join("data", f"{dname}_y_test.npy")

        if os.path.exists(train_path):
            pred_train = np.load(train_path)
            pred_test  = np.load(test_path)
            y_train    = np.load(y_train_path)
            y_test     = np.load(y_test_path)
            print(f"  → Matrizes reais carregadas de T2.")
        else:
            # Mock: estrutura idêntica à real (autonomia conforme contrato)
            n_train = 283 if dname == "finnish" else 43
            n_test  = 122 if dname == "finnish" else 19
            pred_train = matrix_mock(n_train)
            pred_test  = matrix_mock(n_test, random_state=RANDOM_STATE + 99)
            rng_t = np.random.default_rng(RANDOM_STATE)
            y_train = rng_t.uniform(0, 1, size=n_train)
            y_test  = rng_t.uniform(0, 1, size=n_test)
            print(f"  → Usando matrizes MOCK (T2 ainda não executado).")

        result = run_ses_ga(
            pred_matrix_train=pred_train,
            y_train=y_train,
            pred_matrix_test=pred_test,
            y_test=y_test,
        )
        all_results[dname] = result

        # Salva resultado (converte tipos numpy para serializáveis)
        def _to_serializable(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj

        import copy
        result_clean = json.loads(
            json.dumps(result, default=_to_serializable)
        )
        out_path = os.path.join(RESULTS_DIR, f"{dname}_ses_ga.json")
        with open(out_path, "w") as f:
            json.dump(result_clean, f, indent=2)
        print(f"  → Resultado salvo: {out_path}")

    print("\n[T3] ✓ SES-GA concluído para todos os datasets!")
    return all_results


if __name__ == "__main__":
    run_ses_ga_pipeline()
