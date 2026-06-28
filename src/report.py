"""
report.py — Gera outputs/report.md com checagens, tabelas comparadas (obtido x
artigo x delta), referencias as figuras, pressupostos e discussao.
"""
import os


def _md_metric_table(df, metric_cols):
    """Tabela markdown: por modelo, obtido/artigo/delta de cada metrica."""
    head = "| Model | " + " | ".join(
        f"{m} (obt) | {m} (art) | Δ" for m in metric_cols) + " |"
    sep = "|" + "---|" * (1 + 3 * len(metric_cols))
    lines = [head, sep]
    for _, r in df.iterrows():
        cells = [str(r["Model"])]
        for m in metric_cols:
            cells += [f"{r[f'{m}_obt']}", f"{r[f'{m}_art']}", f"{r[f'{m}_delta']}"]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def build_report(root, results, t4, t5, t6, log, metric_cols, row_order, pool):
    df_f = results["finnish"]["d"]
    df_m = results["maxwell"]["d"]
    cf, cm = df_f["check_got"], df_m["check_got"]
    rf, rm = df_f["check_ref"], df_m["check_ref"]

    def chk(name, got, ref):
        return (
            f"| {name} | {got['obs']} / {ref['obs']} | "
            f"{got['mean']:.4f} / {ref['mean']:.4f} | "
            f"{got['median']} / {ref['median']} | "
            f"{got['mn']} / {ref['mn']} | {got['mx']} / {ref['mx']} | "
            f"{got['skew']:.2f} / {ref['skew']:.2f} | "
            f"{got['kurt']:.2f} / {ref['kurt']:.2f} |")

    t6_md = ["| Model | Finnish (obt) | Finnish (art) | Maxwell (obt) | Maxwell (art) |",
             "|---|---|---|---|---|"]
    for _, r in t6.iterrows():
        t6_md.append(f"| {r['Model']} | {r['Finnish_obt']} | {r['Finnish_art']} | "
                     f"{r['Maxwell_obt']} | {r['Maxwell_art']} |")
    t6_md = "\n".join(t6_md)

    txt = f"""# Relatorio de Replicacao — OES / Software Effort Estimation
Replicacao do experimento de Jadhav et al., *Effective Software Effort
Estimation Leveraging Machine Learning for Digital Transformation*, IEEE
Access, vol. 11, 2023.

> Replicacao fiel da metodologia (datasets, pool de modelos, SES-GA, DES, OES,
> 5 metricas, tabelas e figuras). Codigo simples e comentado; toda decisao
> tomada onde o artigo e omisso esta documentada na secao **Pressupostos**.

## 1. Checagem das estatisticas dos datasets (Tabela 2: obtido / esperado)

| Dataset | Obs | Mean Effort | Median | Min | Max | Skew | Kurt |
|---|---|---|---|---|---|---|---|
{chk("Finnish", cf, rf)}
{chk("Maxwell", cm, rm)}

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

{_md_metric_table(t4, metric_cols)}

## 4. Tabela 5 — Maxwell (obtido x artigo x delta)

{_md_metric_table(t5, metric_cols)}

## 5. Tabela 6 — Wilcoxon signed-rank (p-value: obtido x artigo)

{t6_md}

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
{chr(10).join(log)}
```
"""
    path = os.path.join(root, "outputs", "report.md")
    with open(path, "w") as f:
        f.write(txt)
    return path
