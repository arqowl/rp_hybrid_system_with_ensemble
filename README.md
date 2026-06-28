# Replicação — Omni-Ensemble Selection (OES) / Software Effort Estimation

Replicação fiel do experimento de Jadhav et al., *"Effective Software Effort
Estimation Leveraging Machine Learning for Digital Transformation"* (IEEE Access,
vol. 11, 2023): bases **Finnish** e **Maxwell**, pool de 12 regressores
(Tabela 3), seleção de ensembles **SES-GA → DES → OES**, 5 métricas (Eqs. 2–6) e
as Figuras 4–7.

## Como executar

Há duas formas equivalentes (mesma lógica em `src/pipeline.py`, sem duplicação):

**1. Notebooks (recomendado, fase a fase):**
```bash
pip install -r requirements.txt
jupyter notebook notebooks/01_replicacao_OES.ipynb   # ARTIGO: Finnish + Maxwell (replicação fiel)
jupyter notebook notebooks/02_multibase_OES.ipynb    # MULTI: 6 bases de SEE (4 novas + Finnish + Maxwell)
```

**2. Linha de comando (gera tudo de uma vez):**
```bash
pip install -r requirements.txt
python src/run_experiment.py
```

Ambos produzem os mesmos artefatos em `outputs/`. Sementes fixas
(`random_state = 42`) garantem reprodutibilidade.

## Estrutura

```
data/raw/            Finnish407.csv, maxwell.arff (artigo) +
                     desharnais.csv, china.arff, kitchenham.arff, coc81-dem.arff (multi-base)
notebooks/
  01_replicacao_OES.ipynb   ARTIGO: Finnish, depois Maxwell (replicação fiel, por fase)
  02_multibase_OES.ipynb    MULTI: 6 bases de SEE (4 novas + Finnish + Maxwell)
src/
  data_prep.py       Fase 1: carga, ausentes, checagem Tabela 2, seleção 31/21, MinMax, split 70:30
  models.py          Pool (Tabela 3). XGBoost implementado na mão; NB adaptado p/ regressão
  ensembles.py       Fase 2/3: SES-GA, DES (implementado na mão), OES
  metrics.py         sMAPE, MRE, MASE, NSE, COD (Eqs. 2–6)
  tabelas.py         Valores-alvo do artigo + montagem das tabelas (obtido×artigo×Δ)
  figures.py         Figuras 4/5 (scatter) e 6/7 (radar)
  pipeline.py        Funções por fase (fonte única, usada pelos notebooks e CLIs)
  report.py          Gera outputs/report.md
  run_experiment.py  Ponto de entrada por linha de comando (artigo)
  data_prep_v2.py / run_experiment_v2.py   Variante de pré-processamento melhorado (artigo)
  diagnostico_preproc.py   Diagnóstico de pré-processamento (artigo)
  data_multi.py            Carga generalizada multi-base (baseline + melhorado por regras)
  run_experiment_multi.py  Roda o pipeline nas 6 bases (holdout + robustez)
outputs/
  tables/            tabela4_finnish.csv, tabela5_maxwell.csv, tabela6_wilcoxon.csv
  figures/           fig4..fig7 (.png)
  report.md          Relatório: checagens, tabelas comparadas, pressupostos, discussão
  v2/                Variante melhorada do artigo (tabelas, figuras, report_v2.md)
  multi/             Experimento multi-base (tables/, figures/, report_multi.md)
```

## Diagnóstico e variante de pré-processamento melhorado (v2)

Além da replicação fiel, há dois extras (que **não** alteram a replicação):

- **Diagnóstico de pré-processamento:** `python src/diagnostico_preproc.py`
  gera `outputs/diagnostico_preprocessamento.md` + figuras em
  `outputs/figures/diagnostico/` (assimetria do alvo, variância quase-nula,
  redundância/VIF, informação mútua, PCA, outliers).
- **Variante melhorada (v2):** `python src/run_experiment_v2.py` mantém a mesma
  arquitetura (pool → SES-GA → DES → OES) e muda só o pré-processamento
  (log1p no alvo; remoção direcionada de atributos; split estratificado + CV
  repetida). Gera `outputs/v2/` com tabelas comparando **v2 × v1 × artigo** e
  uma tabela de **robustez** (média ± dp em validação cruzada repetida). Use
  `data_prep_v2.py` em conjunto com `pipeline.py` (mesmas fases da replicação).

## Experimento multi-base (6 bases de SEE)

`python src/run_experiment_multi.py` aplica **o mesmo pipeline** (pool → SES-GA →
DES → OES → 5 métricas → Wilcoxon) a 8 bases, em dois modos de pré-processamento
(`baseline` = limpeza mínima; `melhorado` = regras automáticas: remove
variância≈0, correlação>0.98 e informação-mútua≈0, aplica `log1p` no alvo
assimétrico e split estratificado). Inclui um **teste pareado método×método**
(Wilcoxon nos erros absolutos) que, ao rejeitar H0, indica **quem vence** (não só
"rejeita"). Acrescente `robustez` para a validação cruzada repetida:

```bash
python src/run_experiment_multi.py                       # holdout + figuras + ranking + significância + report
python src/run_experiment_multi.py robustez              # robustez (CV repetida) das 8 bases
python src/run_experiment_multi.py robustez debutanizer,abalone   # só as bases grandes (cap=700)
python src/run_experiment_multi.py report_robustez       # (re)escreve a seção de robustez no report
```

Gera `outputs/multi/` (tabelas comparativas por base, `oes_ranking.csv`,
`significancia_pareada.csv`, `robustez_oes.csv`, scatters e `report_multi.md`).
O notebook `02_multibase_OES.ipynb` cobre o mesmo experimento.

**Bases, categoria e procedência (sem duplicatas):**
- **SEE (esforço de software):** Finnish e Maxwell (artigo); **Desharnais**
  (J.M. Desharnais, 1989; 81 proj. canadenses; PROMISE); **China** (499 proj.
  por pontos de função; PROMISE/ISBSG); **Kitchenham** (Kitchenham, Pfleeger,
  McColl & Eagan, 2002; PROMISE); **COCOMO81** (B. Boehm, *Software Engineering
  Economics*, 1981; 63 proj.; PROMISE).
- **Regressão genérica (não-SEE), incluídas a pedido como contraste:**
  `phpWT77lf.arff` = **debutanizer** (sensor virtual de processo químico;
  Fortuna et al., 2007; OpenML) e **abalone** (idade de moluscos; UCI, Nash et
  al., 1994). Nelas os 3 métodos atingem sMAPE baixo (~15–28%), evidenciando que
  o sMAPE alto em SEE é intrínseco à dificuldade do problema, não do método.

Colunas de **vazamento** (derivadas do esforço ou só conhecidas ao fim) são
removidas nos dois modos: China `PDR_*`/`NPD*`/`N_effort`; `Duration`/`Length`/
`Time`/`Actual.duration`; COCOMO `months`/`defects`; além de IDs, datas e strings.

## Observações importantes

- **Datasets validados contra a Tabela 2** (Finnish 405 obs. após remover as 2
  linhas com Worksup=0; Maxwell 62 obs.) — ver `outputs/report.md`.
- **XGBoost é implementado na mão** (`models.XGBoostRegressor`): gradient
  boosting com peso de folha regularizado e ganho de split pela objetivo
  regularizada (a matemática do XGBoost), com os parâmetros do artigo.
- **DES também é implementação própria** (competência local por vizinhos), não
  usa biblioteca externa.
- **CatBoost é a única substituição por biblioteca** (sem rede no container):
  `GradientBoostingRegressor` com os mesmos hiperparâmetros — reimplementar
  ordered boosting + oblivious trees seria complexo e contra "código simples".
- **Decisões onde o artigo é omisso** (seleção exata de features, hiperparâmetros
  do GA, neurônios do MLP, kernel/k de SVM/kNN, etc.) estão na seção
  **"Pressupostos"** do `outputs/report.md`.
- Os **valores decimais exatos das Tabelas 4–6 não são reproduzíveis** dadas as
  ambiguidades do artigo; o relatório traz obtido × artigo × delta.
