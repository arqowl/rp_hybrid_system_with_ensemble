"""
ensembles.py — Fases 2 e 3 do artigo.

SES-GA : Static Ensemble Selection via Algoritmo Genetico.
         Cromossomo = vetor binario sobre o pool (inclui/exclui cada modelo).
         Fitness   = R^2 (quadrado do r de Pearson, Eq.1) da media das
                     predicoes dos modelos selecionados no conjunto de treino.
         Saida     = subconjunto de modelos selecionados + predicoes (media).

DES    : Dynamic Ensemble Selection. Para cada caso de teste, mede a
         competencia local de cada modelo selecionado pelo SES-GA usando os
         k vizinhos mais proximos no espaco de features (competencia = -|erro|
         medio local) e prediz com o(s) modelo(s) localmente mais competente(s).

OES    : Omni-Ensemble Selection (proposto). Combina SES e DES pela media das
         duas predicoes. E a abordagem "Proposed" das Tabelas 4/5.

Hiperparametros do GA nao especificados no artigo -> defaults simples e fixos
(pop=30, geracoes=60, crossover=0.8, mutacao=0.1). Ver 'Pressupostos'.
"""
import numpy as np

GA_POP = 30
GA_GEN = 60
GA_CX = 0.8
GA_MUT = 0.1
GA_LAMBDA = 0.5   # peso da DIVERSIDADE na fitness: fitness = acuracia + lambda*div.
                  # O artigo diz que o SES-GA otimiza "accuracy AND diversity"
                  # de forma colaborativa; e a diversidade (nao um piso de
                  # cardinalidade) que desencoraja o ensemble colapsar em 1
                  # modelo. [Pressupostos]
MIN_MODELS = 1    # apenas garante ensemble NAO-VAZIO (validade), nao um piso.
DES_K = 5          # vizinhos para competencia local


def _r2_fitness(y, yhat):
    """R^2 = (r de Pearson)^2, Eq.1 do artigo. Penaliza casos degenerados."""
    if np.std(yhat) == 0 or np.std(y) == 0:
        return 0.0
    r = np.corrcoef(y, yhat)[0, 1]
    return r ** 2


def _ensemble_pred(preds, mask):
    """Media das predicoes dos modelos onde mask==1."""
    idx = np.where(mask == 1)[0]
    if len(idx) == 0:
        return None
    return preds[idx].mean(axis=0)


def _repair(c, M, rng):
    """Garante que o cromossomo tenha pelo menos MIN_MODELS modelos ativos."""
    falta = MIN_MODELS - int(c.sum())
    if falta > 0:
        zeros = np.where(c == 0)[0]
        add = rng.choice(zeros, size=min(falta, len(zeros)), replace=False)
        c[add] = 1
    return c


def _diversity(preds, mask, y):
    """Diversidade = 1 - correlacao media par-a-par entre os ERROS (residuos)
    dos modelos selecionados. Modelos que erram juntos -> alta correlacao ->
    baixa diversidade (redundantes). Modelos complementares -> alta diversidade.

    Para |S| < 2 nao ha pares -> diversidade = 0 (singleton nao recebe bonus,
    e por isso tende a perder para conjuntos diversos de acuracia parecida).
    """
    idx = np.where(mask == 1)[0]
    if len(idx) < 2:
        return 0.0
    err = preds[idx] - y[None, :]                 # (S, n) residuos OOF
    keep = np.std(err, axis=1) > 0                # evita NaN no corrcoef
    err = err[keep]
    if len(err) < 2:
        return 0.0
    C = np.corrcoef(err)                          # matriz SxS de correlacoes
    iu = np.triu_indices(len(err), k=1)           # pares (a<b)
    return 1.0 - float(np.mean(C[iu]))


def ses_ga(train_preds, y_train, rng):
    """Roda o GA e devolve (mascara_binaria, indices_selecionados).

    Fitness = acuracia + GA_LAMBDA * diversidade (otimizacao colaborativa de
    acuracia E diversidade, como descrito no artigo):
      - acuracia    = R^2 (quadrado do r de Pearson, Eq.1) da media das
                      predicoes dos modelos selecionados;
      - diversidade = 1 - correlacao media par-a-par entre os erros deles.

    train_preds: array (n_modelos, n_amostras_treino) com predicoes OOF de cada
                 modelo do pool (validacao do GA).
    """
    M = train_preds.shape[0]

    def fitness(mask):
        yp = _ensemble_pred(train_preds, mask)
        if yp is None:
            return -1.0
        acc = _r2_fitness(y_train, yp)
        div = _diversity(train_preds, mask, y_train)
        return acc + GA_LAMBDA * div

    # populacao inicial aleatoria (com cardinalidade minima garantida)
    pop = rng.integers(0, 2, size=(GA_POP, M))
    pop = np.array([_repair(c, M, rng) for c in pop])
    fits = np.array([fitness(c) for c in pop])

    for _ in range(GA_GEN):
        new = []
        while len(new) < GA_POP:
            # selecao por torneio (2 candidatos)
            a, b = rng.integers(0, GA_POP, 2)
            p1 = pop[a] if fits[a] >= fits[b] else pop[b]
            a, b = rng.integers(0, GA_POP, 2)
            p2 = pop[a] if fits[a] >= fits[b] else pop[b]
            # crossover de 1 ponto
            child = p1.copy()
            if rng.random() < GA_CX:
                pt = rng.integers(1, M)
                child = np.concatenate([p1[:pt], p2[pt:]])
            # mutacao bit-flip
            for j in range(M):
                if rng.random() < GA_MUT:
                    child[j] = 1 - child[j]
            new.append(_repair(child, M, rng))
        pop = np.array(new)
        fits = np.array([fitness(c) for c in pop])

    best = pop[int(np.argmax(fits))]
    return best, np.where(best == 1)[0]


def des_predict(sel_idx, train_preds, test_preds, X_train, X_test, y_train,
                k_sub=3):
    """DES: para cada caso de teste escolhe um SUBCONJUNTO dos modelos
    localmente mais competentes (top-k) e prediz pela media deles.

    Competencia = menor erro absoluto medio nos DES_K vizinhos de treino.
    O artigo descreve o DES selecionando "um subconjunto" por amostra; usar a
    media dos top-k competentes (em vez de 1 unico) e mais fiel e menos volatil.
    sel_idx = indices (no pool) dos modelos selecionados pelo SES-GA.
    """
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=min(DES_K, len(X_train))).fit(X_train)
    _, neigh = nn.kneighbors(X_test)

    abs_err = np.abs(train_preds[sel_idx] - y_train[None, :])  # (S, n_train)
    # subconjunto proprio e dinamico: por padrao, descarta por amostra o modelo
    # localmente menos competente (mantem os demais). Garante DES != SES mesmo
    # com poucos modelos selecionados, sem colapsar tudo num unico modelo.
    ksub = min(k_sub, max(1, len(sel_idx) - 1))

    out = np.empty(test_preds.shape[1])
    for i in range(test_preds.shape[1]):
        local = abs_err[:, neigh[i]].mean(axis=1)        # erro local por modelo
        order = np.argsort(local)[:ksub]                  # top-k competentes
        chosen = sel_idx[order]
        out[i] = test_preds[chosen, i].mean()
    return out


def oes_predict(ses_pred, des_pred):
    """OES = combinacao (media) de SES e DES."""
    return (np.asarray(ses_pred) + np.asarray(des_pred)) / 2.0
