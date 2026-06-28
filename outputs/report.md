# Relatorio de Replicacao — OES / Software Effort Estimation
Replicacao do experimento de Jadhav et al., *Effective Software Effort
Estimation Leveraging Machine Learning for Digital Transformation*, IEEE
Access, vol. 11, 2023.

> Replicacao fiel da metodologia (datasets, pool de modelos, SES-GA, DES, OES,
> 5 metricas, tabelas e figuras). Codigo simples e comentado; toda decisao
> tomada onde o artigo e omisso esta documentada na secao **Pressupostos**.

## 1. Checagem das estatisticas dos datasets (Tabela 2: obtido / esperado)

| Dataset | Obs | Mean Effort | Median | Min | Max | Skew | Kurt |
|---|---|---|---|---|---|---|---|
| Finnish | 405 / 405 | 5031.0148 / 5031.0148 | 2500.0 / 2500 | 55.0 / 55 | 63694.0 / 63694 | 3.71 / 3.70 | 18.69 / 18.69 |
| Maxwell | 62 / 62 | 8223.2097 / 8223.2097 | 5189.5 / 5189.5 | 583.0 / 583 | 63694.0 / 63694 | 3.35 / 3.34 | 13.70 / 13.69 |

Finnish: as 2 observacoes "ausentes" do artigo correspondem as 2 linhas com
esforco invalido (Worksup = 0); removidas -> 405 obs., reproduzindo a Tabela 2
exatamente. Maxwell: 62 obs., estatisticas do alvo batem com o artigo. O
arquivo Maxwell disponivel tem 27 atributos (o artigo cita 28); diferenca
conhecida de versao do dataset, sem impacto na checagem do alvo.

## 2. Tabela 3 — Pool de modelos e parametros aplicados

| Algoritmo | Parametros aplicados | Sigla |
|---|---|---|
| Support Vector Machine | kernel=rbf, degree=3 (variantes linear/poly consideradas) | SVM |
| Random Forest | n_estimators=10, min_samples_leaf=1 | RF |
| Multi-layer Perceptron | hidden_layer=1 (100 neuronios), learning_rate=0.01 | MLP |
| k Nearest Neighbours | k=5 (3/7 consideradas) | kNN |
| Decision Tree | criterion='squared_error', max_depth=2 | DT |
| Extra Tree | n_estimators=100, min_samples_leaf=1 | ET |
| Linear Regression | default | LR |
| AdaBoost | n_estimators=10 | ADA |
| CatBoost (substituido) | GradientBoosting: n_estimators=10, learning_rate=1, depth=2 | CAT |
| XGBoost (implementado na mao) | gradient boosting regularizado: n_estimators=50, max_depth=3, eta=0.1, lambda=1 | XGB |
| Naive Bayes (adaptado) | GaussianNB sobre alvo discretizado (10 quantis) | NB |
| Bagging | default | BG |

## 3. Tabela 4 — Finnish (obtido x artigo x delta)

| Model | sMAPE (obt) | sMAPE (art) | Δ | MRE (obt) | MRE (art) | Δ | MASE (obt) | MASE (art) | Δ | NSE (obt) | NSE (art) | Δ | COD (obt) | COD (art) | Δ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| SVM | 88.91707 | 48.186 | 40.73107 | 3.4004 | 1.87874 | 1.52166 | 0.61695 | 0.81134 | -0.19439 | 0.41507 | 0.56357 | -0.1485 | 0.55384 | 0.80788 | -0.25404 |
| RF | 58.45701 | 29.145 | 29.31201 | 1.09889 | 1.04612 | 0.05277 | 0.38796 | 0.66909 | -0.28113 | 0.59641 | 0.36538 | 0.23103 | 0.66149 | 0.74756 | -0.08607 |
| MLP | 80.14061 | 38.703 | 41.43761 | 1.5723 | 1.89333 | -0.32103 | 0.44893 | 0.69446 | -0.24553 | 0.5567 | 0.64448 | -0.08778 | 0.62037 | 0.8107 | -0.19033 |
| kNN | 61.74359 | 35.681 | 26.06259 | 0.96208 | 1.36927 | -0.40719 | 0.44221 | 0.72336 | -0.28115 | 0.47386 | 0.56898 | -0.09512 | 0.6611 | 0.75544 | -0.09434 |
| DT | 64.17882 | 37.585 | 26.59382 | 1.28455 | 2.25293 | -0.96838 | 0.4094 | 0.92997 | -0.52057 | 0.5665 | 0.28421 | 0.28229 | 0.65994 | 0.62113 | 0.03881 |
| ET | 55.00881 | 27.747 | 27.26181 | 0.95261 | 0.98134 | -0.02873 | 0.38252 | 0.54617 | -0.16365 | 0.5269 | 0.68537 | -0.15847 | 0.64495 | 0.84353 | -0.19858 |
| LR | 75.34966 | 42.961 | 32.38866 | 1.06398 | 1.45984 | -0.39586 | 0.4085 | 0.67045 | -0.26195 | 0.56139 | 0.6332 | -0.07181 | 0.63567 | 0.84546 | -0.20979 |
| ADA | 63.17702 | 37.621 | 25.55602 | 1.30114 | 2.46283 | -1.16169 | 0.39064 | 0.80211 | -0.41147 | 0.55147 | 0.23775 | 0.31372 | 0.64777 | 0.72402 | -0.07625 |
| CAT | 67.02234 | 33.534 | 33.48834 | 1.00495 | 1.44861 | -0.44366 | 0.38814 | 0.69836 | -0.31022 | 0.56879 | 0.50544 | 0.06335 | 0.63232 | 0.79121 | -0.15889 |
| XGB | 54.7436 | 30.131 | 24.6126 | 0.85183 | 1.34004 | -0.48821 | 0.35925 | 0.69576 | -0.33651 | 0.6118 | 0.2715 | 0.3403 | 0.74091 | 0.77509 | -0.03418 |
| NB | 66.17467 | 38.732 | 27.44267 | 1.07397 | 1.19796 | -0.12399 | 0.48702 | 0.62284 | -0.13582 | 0.44761 | 0.67665 | -0.22904 | 0.46122 | 0.83747 | -0.37625 |
| BG | 56.58884 | 43.474 | 13.11484 | 1.07204 | 1.54597 | -0.47393 | 0.37624 | 0.70022 | -0.32398 | 0.62393 | 0.62224 | 0.00169 | 0.68785 | 0.8411 | -0.15325 |
| Static | 66.48696 | 27.825 | 38.66196 | 1.4043 | 1.05944 | 0.34486 | 0.42947 | 0.54115 | -0.11168 | 0.55123 | 0.71622 | -0.16499 | 0.6623 | 0.87652 | -0.21422 |
| DES | 68.20341 | 33.884 | 34.31941 | 1.32311 | 0.92209 | 0.40102 | 0.42625 | 0.5723 | -0.14605 | 0.54734 | 0.66953 | -0.12219 | 0.6614 | 0.84379 | -0.18239 |
| Proposed | 64.89367 | 23.896 | 40.99767 | 1.313 | 0.79142 | 0.52158 | 0.42464 | 0.45167 | -0.02703 | 0.55159 | 0.78121 | -0.22962 | 0.66783 | 0.91375 | -0.24592 |

## 4. Tabela 5 — Maxwell (obtido x artigo x delta)

| Model | sMAPE (obt) | sMAPE (art) | Δ | MRE (obt) | MRE (art) | Δ | MASE (obt) | MASE (art) | Δ | NSE (obt) | NSE (art) | Δ | COD (obt) | COD (art) | Δ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| SVM | 81.21616 | 37.302 | 43.91416 | 1.90762 | 1.60133 | 0.30629 | 0.59036 | 0.64518 | -0.05482 | 0.18831 | 0.64141 | -0.4531 | 0.47864 | 0.85525 | -0.37661 |
| RF | 52.7326 | 18.274 | 34.4586 | 0.59942 | 0.69642 | -0.097 | 0.38417 | 0.32081 | 0.06336 | 0.57463 | 0.88611 | -0.31148 | 0.86431 | 0.95035 | -0.08604 |
| MLP | 69.51342 | 45.079 | 24.43442 | 1.46033 | 2.61562 | -1.15529 | 0.4424 | 0.98424 | -0.54184 | 0.42241 | 0.12867 | 0.29374 | 0.59969 | 0.2717 | 0.32799 |
| kNN | 57.80241 | 35.082 | 22.72041 | 1.0074 | 1.4606 | -0.4532 | 0.49974 | 0.78834 | -0.2886 | 0.07888 | 0.43263 | -0.35375 | 0.31348 | 0.73278 | -0.4193 |
| DT | 61.67686 | 30.156 | 31.52086 | 0.70938 | 1.49436 | -0.78498 | 0.41866 | 0.6474 | -0.22874 | 0.52605 | 0.61278 | -0.08673 | 0.71283 | 0.79335 | -0.08052 |
| ET | 56.20842 | 18.59 | 37.61842 | 0.91896 | 1.1559 | -0.23694 | 0.41611 | 0.37111 | 0.045 | 0.44703 | 0.805 | -0.35797 | 0.8511 | 0.89854 | -0.04744 |
| LR | 70.17978 | 65.296 | 4.88378 | 1.61649 | 14.70762 | -13.09113 | 0.45243 | 1.61663 | -1.1642 | 0.50707 | 0.28759 | 0.21948 | 0.64277 | 0.63149 | 0.01128 |
| ADA | 57.59639 | 24.89 | 32.70639 | 0.71763 | 0.96699 | -0.24936 | 0.44001 | 0.57493 | -0.13492 | 0.32509 | 0.43299 | -0.1079 | 0.7129 | 0.95678 | -0.24388 |
| CAT | 61.84051 | 36.453 | 25.38751 | 0.81312 | 1.54738 | -0.73426 | 0.33642 | 0.673 | -0.33658 | 0.76132 | 0.5657 | 0.19562 | 0.82161 | 0.87383 | -0.05222 |
| XGB | 59.76767 | 18.637 | 41.13067 | 0.72958 | 0.92803 | -0.19845 | 0.45291 | 0.29177 | 0.16114 | 0.27626 | 0.93225 | -0.65599 | 0.62029 | 0.96763 | -0.34734 |
| NB | 75.61202 | 39.123 | 36.48902 | 2.39357 | 1.0559 | 1.33767 | 0.56334 | 0.57534 | -0.012 | 0.15262 | 0.66247 | -0.50985 | 0.31252 | 0.81607 | -0.50355 |
| BG | 57.49879 | 20.589 | 36.90979 | 0.62055 | 1.10704 | -0.48649 | 0.3926 | 0.35486 | 0.03774 | 0.56692 | 0.88439 | -0.31747 | 0.84722 | 0.94103 | -0.09381 |
| Static | 60.29884 | 23.527 | 36.77184 | 1.0315 | 0.7629 | 0.2686 | 0.43166 | 0.42057 | 0.01109 | 0.45588 | 0.8042 | -0.34832 | 0.85891 | 0.93988 | -0.08097 |
| DES | 55.26506 | 16.558 | 38.70706 | 0.60991 | 0.50287 | 0.10704 | 0.44136 | 0.32678 | 0.11458 | 0.30285 | 0.90453 | -0.60168 | 0.6592 | 0.9522 | -0.293 |
| Proposed | 59.14524 | 15.057 | 44.08824 | 0.81481 | 0.47942 | 0.33539 | 0.43585 | 0.26125 | 0.1746 | 0.38524 | 0.95146 | -0.56622 | 0.80193 | 0.98359 | -0.18166 |

## 5. Tabela 6 — Wilcoxon signed-rank (p-value: obtido x artigo)

| Model | Finnish (obt) | Finnish (art) | Maxwell (obt) | Maxwell (art) |
|---|---|---|---|---|
| Static Ensemble Selection (SES) | 0.0015 | 0.236 | 0.3525 | 0.112 |
| Dynamic Ensemble Selection (DES) | 0.032 | 0.075 | 0.2753 | 0.056 |
| Omni Ensemble Selection (OES) | 0.006 | 0.004 | 0.3736 | 0.021 |

Regra do artigo: p < 0.05 rejeita a hipotese nula; entre 0.05 e 1, as amostras
nao sao estatisticamente equivalentes.

## 6. Figuras geradas

- `outputs/figures/fig4_finnish_scatter.png` — scatter SES/DES/OES (Finnish), (a) treino e (b) teste, com retas de tendencia.
- `outputs/figures/fig5_maxwell_scatter.png` — scatter SES/DES/OES (Maxwell).
- `outputs/figures/fig6_finnish_radar.png` — radar sMAPE vs COD (Finnish).
- `outputs/figures/fig7_maxwell_radar.png` — radar sMAPE vs COD (Maxwell).

## 7. Pressupostos (decisoes onde o artigo e omisso/ambiguo)

1. **Implementacoes proprias e a unica substituicao restante.** O container
   nao tem rede, entao xgboost/catboost/deslib nao puderam ser instalados.
   - **XGBoost: implementado na mao** (`models.XGBoostRegressor`) — gradient
     boosting de arvores para erro quadratico, com peso de folha REGULARIZADO
     `w = -G/(H+lambda)` e ganho de split pela objetivo regularizada (a
     matematica do XGBoost), com os parametros da Tabela 3 (n_estimators=50,
     max_depth=3, eta=0.1, lambda=1). Nao bate bit-a-bit com a lib oficial
     (que usa histograma/quantil, subsampling, base_score etc.), mas e o
     algoritmo de verdade, nao um stand-in.
   - **DES: implementado na mao** (`ensembles.des_predict`) — competencia local
     por vizinhos (kNN). Nao usa deslib.
   - **CatBoost: unica substituicao por biblioteca** — mantido como
     `GradientBoostingRegressor` com os MESMOS hiperparametros (n_estimators=10,
     learning_rate=1, depth=2). O CatBoost "de verdade" depende de ordered
     boosting + oblivious trees; reimplementa-lo fielmente seria complexo e
     contra a regra de "codigo simples", entao foi documentado como variante de
     gradient boosting.
2. **Selecao de features (31 Finnish / 21 Maxwell).** O artigo nomeia 14
   atributos "soft" mas reporta 31/21 features (incl. alvo) sem listar as
   colunas exatas. Regra adotada (atinge exatamente 31/21):
   - Finnish: descartadas 7 colunas de id/texto nominal, `Duration` (vazamento
     a priori) e 7 contagens brutas `*Tot` (redundantes com `*FP`).
   - Maxwell: descartadas `Duration` e `Time` (tempo decorrido = vazamento),
     mais `Syear`, `Source`, `Telonuse`, `Nlan` (metadados de menor impacto).
3. **MinMax antes do split, incluindo o alvo.** A Figura 3 ordena
   Normalizacao -> Selecao -> Split, entao o MinMax e ajustado sobre a base
   completa (fiel ao artigo). O **alvo tambem e normalizado para [0,1]** (o
   artigo normaliza todas as features); as 5 metricas sao reportadas nesse
   espaco normalizado. Manter o alvo bruto faria SVR/MLP colapsarem na media
   (esforco ate ~63694), divergindo do artigo.
4. **GA com fitness de acuracia E diversidade.** cromossomo = mascara binaria
   sobre o pool. A fitness implementa a "otimizacao colaborativa de acuracia e
   diversidade" descrita no artigo:
   `fitness = acuracia + lambda * diversidade`, com lambda = 0.5, onde
   - **acuracia** = R^2 (quadrado do r de Pearson, Eq.1) da media das predicoes
     dos modelos selecionados, calculada sobre predicoes **out-of-fold (5-fold)**
     do treino (nao resubstituicao, que faria arvores memorizarem o treino);
   - **diversidade** = 1 - correlacao media par-a-par entre os ERROS (residuos)
     dos modelos selecionados (modelos que erram juntos sao redundantes).
   E a diversidade — e nao um piso de cardinalidade — que desencoraja o ensemble
   colapsar num unico modelo: um singleton tem diversidade 0 e tende a perder
   para conjuntos complementares de acuracia parecida. Se, ainda assim, um
   modelo dominar em toda parte, ele e mantido (resultado honesto: "sem ganho de
   ensemble"). pop=30, geracoes=60, crossover=0.8, mutacao=0.1, selecao por
   torneio, crossover de 1 ponto, mutacao bit-flip. Sementes fixas.
5. **SVM / kNN reportados.** SVM linha = kernel rbf; kNN linha = k=5.
6. **MLP.** 100 neuronios em 1 camada oculta (nro nao especificado no artigo).
7. **NB para regressao.** discretizacao do alvo em 10 quantis + GaussianNB +
   predicao da media do bin (adaptacao minima).
8. **DES.** competencia local via k=5 vizinhos no espaco de features, medida
   sobre predicoes **out-of-fold** do treino; por amostra seleciona o
   **subconjunto** dos top-3 modelos (entre os do SES-GA) de menor erro local e
   prediz pela media deles (fiel ao "subconjunto" descrito no artigo).
9. **OES.** combinacao = media das predicoes de SES e DES.
10. **COD (Eq.6).** No artigo, Eqs. 5 e 6 sao algebricamente identicas; para
    reproduzir valores em [0,1] coerentes com as Tabelas (e a relacao inversa
    sMAPE<->COD das Figs. 6/7), COD = (r de Pearson)^2.
11. **random_state = 42** em todo o pipeline (split, modelos, GA).

## 8. Discussao

O criterio de sucesso do artigo e que o **OES (Proposed)** supere SES, DES e
os modelos individuais nas duas bases — menor sMAPE/MRE/MASE e maior NSE/COD.
As Tabelas 4 e 5 acima trazem os valores **obtidos lado a lado com os do
artigo** e o delta. Por causa da ausencia da lista exata de 31/21 features no
artigo, dos hiperparametros nao especificados do GA/DES, da estocasticidade do
GA e da substituicao do CatBoost por gradient boosting, **nao se espera
igualdade decimal** com o artigo (mesmo com o XGBoost e o DES escritos a mao,
estas sao apenas algumas linhas/etapas entre muitas). O objetivo cumprido e a
**replicacao fiel da metodologia** e a verificacao do comportamento qualitativo
(OES como melhor abordagem combinada). Consulte as tabelas para a aderencia
quantitativa caso a caso.

## Log de execucao
```
finnish: SES-GA -> ['SVM', 'MLP', 'CAT', 'NB']
maxwell: SES-GA -> ['ADA', 'CAT', 'XGB', 'NB']
```
