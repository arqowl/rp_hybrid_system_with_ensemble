# Sistema Híbrido de Omni-Ensemble com Seleção Multiobjetivo para Estimação de Esforço de Software (SEE)

Projeto da disciplina de **Reconhecimento de Padrões**. Estende a abordagem
*Omni-Ensemble Selection* (OES) de **Jadhav et al., IEEE Access, 2023**
(DOI: 10.1109/ACCESS.2023.3293432), propondo uma alteração na etapa de **seleção** de
modelos: o algoritmo genético mono-objetivo do artigo é substituído por uma formulação
**multiobjetivo (precisão × parcimônia)** resolvida por um NSGA-II implementado do zero.

> **Estado atual em uma linha:** o pipeline roda fim-a-fim e gera resultados reproduzíveis,
> mas o resultado empírico é **nulo** — sobre 4 bases pequenas o método proposto não supera
> os baselines e o teste de Friedman não acusa diferença significativa (p = 0,284). Ver
> [Resultados](#resultados-resumo) e [Limitações](#limitações-conhecidas).

---

## O que o sistema faz (3 etapas do ensemble)

| Etapa | O que faz | Contribuição deste projeto |
|---|---|---|
| **Geração** | Treina um pool de 5 regressores individuais de paradigmas distintos: LR, SVR, MLP (*eager*), kNN (*lazy*), DT (árvore). Nenhum componente é, ele próprio, um ensemble. | Organização do pool por paradigma e justificativa de diversidade. |
| **Seleção** | **SES-GA** (mono-objetivo) e **SES-GA-Multi** (NSGA-II, R² × parcimônia) escolhem o subconjunto de modelos; **DES** seleciona dinamicamente por competência local. | **Contribuição central:** a formulação multiobjetivo da seleção estática via NSGA-II próprio. |
| **Combinação** | Média aritmética simples das predições selecionadas. | *(Ainda não há contribuição aqui — a combinação ponderada por competência é trabalho futuro.)* |

**Validação estatística:** teste de **Friedman** + post-hoc de **Bonferroni-Dunn**
(controle = SES-GA-Multi), conforme Demšar (2006) — mais rigoroso que o Wilcoxon isolado do
artigo original por controlar o erro familiar de comparações múltiplas.

**Implementação própria (não é "só sklearn"):** algoritmo genético, NSGA-II
(não-dominância + *crowding distance*), DES por competência local, combinação e os testes de
Friedman/Bonferroni-Dunn são código próprio em NumPy/SciPy. Apenas os regressores-base vêm de
biblioteca.

---

## Estrutura do projeto

```
.
├── data/
│   ├── raw/                     # bases originais do PROMISE (você precisa colocá-las aqui)
│   │   ├── Finnish407.csv        # presente
│   │   ├── maxwell.arff          # presente
│   │   ├── desharnais.csv        # necessário (ver "Dados")
│   │   ├── cocomo81.arff         # necessário
│   │   └── china.arff            # necessário (5ª base; ausente — ver Limitações)
│   └── processed/               # artefatos gerados pelo pré-processamento (npy, metadata)
├── src/                         # fonte única de verdade (toda a lógica vive aqui)
│   ├── dataset_loader.py         # T1: carga, MinMax, holdout 70/30, OOF, registro das 5 bases
│   ├── train_pool.py             # T2: treina o pool de 5 modelos; gera matrizes de predição
│   ├── ses_ga_single.py          # T3: SES-GA mono-objetivo (R²)
│   ├── ses_ga_multi.py           # T4: SES-GA multiobjetivo via NSGA-II (R² × parcimônia)
│   ├── des_dynamic.py            # T5: seleção dinâmica por competência local
│   ├── combination.py            # combinação (média simples; ponto de extensão p/ competência)
│   ├── evaluator.py              # T6: métricas (sMAPE, MRE, MASE, NSE, COD)
│   ├── statistical_tests.py      # T7: Friedman + Bonferroni-Dunn
│   ├── make_results.py           # T8: gera TODOS os resultados e figuras de forma reprodutível
│   └── telemetria.py             # opcional, NÃO faz parte do experimento (ver Notas)
├── notebooks/
│   └── 01_pipeline_unificado.ipynb  # orquestrador fino: importa src/ e roda T1→T8
├── results/                     # saídas do experimento
│   ├── *_ses_ga.json             # seleção mono-objetivo por base
│   ├── *_ses_ga_multi.json       # frente de Pareto e soluções multiobjetivo por base
│   ├── *_pareto_front.npy
│   ├── teste_friedman.txt        # saída numérica do teste de hipótese
│   └── figuras/                  # 5 figuras (CD, ranks, Pareto, sMAPE por base, tabela)
├── docs/                        # relatório científico (.md) e anotações
├── requirements.txt
└── README.md
```

---

## Requisitos

- **Python 3.10+**
- Dependências (ver `requirements.txt`): `numpy`, `scipy`, `pandas`, `scikit-learn`,
  `matplotlib`, `joblib`, `jupyter`.

```bash
# (recomendado) ambiente virtual
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

---

## Dados

As bases são as **reais do repositório PROMISE / tera-PROMISE** (SEE). **Nenhum dado
sintético é usado** — o carregador falha com mensagem explícita se um arquivo esperado não
estiver em `data/raw/`. Coloque os arquivos com exatamente estes nomes:

| Base | Arquivo esperado em `data/raw/` | Alvo (esforço) | Status no backup |
|---|---|---|---|
| Finnish | `Finnish407.csv` | `worksup` | ✅ presente |
| Maxwell | `maxwell.arff` | `effort` | ✅ presente |
| Desharnais | `desharnais.csv` | `effort` | ⚠️ obter no PROMISE |
| COCOMO81 | `cocomo81.arff` | `actual` | ⚠️ obter no PROMISE |
| China | `china.arff` | `effort` | ❌ ausente (5ª base) |

Origem: repositório PROMISE de *effort estimation*
(ex.: `http://openscience.us/repo/effort/`). Apenas as bases cujos arquivos estiverem
presentes serão processadas; as demais são puladas automaticamente nas etapas seguintes.

> O pré-processamento usa **normalização MinMax [0,1] ajustada só no treino** (sem vazamento)
> e **holdout 70/30 com `random_state = 42`**. A flag `LOG_TARGET` em `dataset_loader.py`
> está em `False` por fidelidade ao artigo (MinMax sobre o esforço cru).

---

## Como rodar

### Opção 1 — Regerar tudo de uma vez (recomendado)

A partir da **raiz do projeto** (para que os imports `from src...` resolvam):

```bash
python -m src.make_results
```

Isso executa o pipeline completo (T1→T8) e grava em `results/`:

- `tabela_metricas.csv` — média por método (sMAPE, MRE, COD) + rank médio;
- `tabela_metricas_por_base.csv` — detalhe por base;
- `teste_friedman.txt` — saída numérica de Friedman + Bonferroni-Dunn;
- `figuras/1_diagrama_diferenca_critica.png`, `2_ranks_medios.png`, `3_frente_pareto.png`,
  `4_smape_por_base.png`, `5_tabela_friedman.png`.

(As figuras antigas em `results/figuras/` são apagadas no início para evitar saída
desatualizada.)

### Opção 2 — Notebook (exploração passo a passo)

```bash
jupyter notebook notebooks/01_pipeline_unificado.ipynb
```

A primeira célula faz o *bootstrap* (coloca a raiz do projeto no `sys.path`). As células
seguem T1 (dados) → T2 (pool) → T3 (SES-GA) → T4 (SES-GA-Multi) → T5–T8 (DES, combinação,
métricas e teste de hipótese).

### Opção 3 — Módulos isolados

Cada módulo de `src/` tem um `if __name__ == "__main__"` para teste isolado, por exemplo:

```bash
python -m src.dataset_loader      # só o pré-processamento
python -m src.train_pool          # só o treino do pool
python -m src.ses_ga_multi        # só a seleção multiobjetivo
```

---

## Resultados (resumo)

Avaliação real sobre **4 bases** (Finnish, Maxwell, Desharnais, COCOMO81), 9 métodos.
Ranqueamento por sMAPE (menor = melhor):

| Método | Rank médio | sMAPE méd. (%) | COD méd. |
|---|---|---|---|
| DES | 3,25 | 70,7 | 0,301 |
| kNN | 3,75 | 66,6 | 0,212 |
| DT | 3,75 | 67,7 | 0,473 |
| Static | 4,00 | 72,5 | 0,438 |
| SES-GA | 4,25 | 75,3 | 0,507 |
| MLP | 6,00 | 85,1 | 0,445 |
| **SES-GA-Multi (proposto)** | **6,00** | **84,3** | **0,446** |
| LR | 6,50 | 85,3 | 0,436 |
| SVR | 7,50 | 84,6 | 0,169 |

**Teste de Friedman (sMAPE, 4×9):** F = 9,73; **p = 0,284 → não significativo**.
**Bonferroni-Dunn:** CD = 5,30; nenhum método difere significativamente do controle.

Leitura honesta: nestas bases pequenas, a seleção multiobjetivo combinada por média **não
supera** os baselines, e como a diferença crítica (5,30) é maior que toda a amplitude de
ranks observada (4,25), o experimento **não tem poder** para distinguir os métodos. O detalhe
e a interpretação estão no relatório em `docs/`.

---

## Limitações conhecidas

1. **Quatro bases, não cinco.** A base **China** (a maior, escolhida justamente para dar
   poder ao Friedman) está ausente. Com 4 bases o teste é fortemente subdimensionado — esta é
   a limitação mais séria e a prioridade nº 1 para reexecução.
2. **Combinação adaptativa não implementada.** Das duas contribuições previstas, só a seleção
   multiobjetivo existe; a combinação ponderada por competência local é um ponto de extensão
   em `combination.py`. Todas as estratégias de ensemble usam média simples.
3. **Métricas parcialmente consolidadas.** Os artefatos consolidam **sMAPE, MRE e COD**;
   MASE e NSE (este idêntico em fórmula ao COD) não entram na tabela-resumo, embora estejam
   implementados em `evaluator.py`.
4. **Resultado empírico nulo.** O valor atual do trabalho é **metodológico** (formulação
   multiobjetivo + validação mais rigorosa), não um ganho de desempenho.
5. **Bases pequenas e assimétricas.** O COCOMO81 domina a escala de sMAPE; por isso a análise
   é feita por **ranks**, não por médias brutas entre bases.

---

## Notas

- **`src/telemetria.py` é opcional e não faz parte do experimento.** Ele importa `torch`,
  `psutil` e `pynvml` (telemetria de hardware/GPU), que **não estão** em `requirements.txt`.
  Nada no pipeline (`make_results.py`, notebook) depende dele; não o execute a menos que
  instale essas dependências à parte.
- **Reprodutibilidade:** semente fixa `42` em todo o pipeline; seleção feita sobre predições
  *out-of-fold* (sem vazamento); hiperparâmetros do NSGA-II registrados em
  `results/*_ses_ga_multi.json` (população 60, 100 gerações, cruzamento 0,8, mutação 0,02,
  torneio 3, elitismo 2). Tempo da seleção multiobjetivo: ~4,4–4,9 s por base. Resultados
  podem variar ligeiramente entre versões de biblioteca; fixe as versões do `requirements.txt`
  para reprodução exata.

---

## Referências

- A. Jadhav, S. K. Shandilya, I. Izonin, M. Gregus. "Effective Software Effort Estimation
  Leveraging Machine Learning for Digital Transformation." *IEEE Access*, vol. 11,
  pp. 83523–83536, 2023. DOI: 10.1109/ACCESS.2023.3293432. *(artigo de referência / objeto de
  extensão)*
- J. Demšar. "Statistical Comparisons of Classifiers over Multiple Data Sets." *JMLR*, vol. 7,
  2006.
- K. Deb, A. Pratap, S. Agarwal, T. Meyarivan. "A Fast and Elitist Multiobjective Genetic
  Algorithm: NSGA-II." *IEEE Trans. Evolutionary Computation*, vol. 6, no. 2, 2002.
- A. S. Britto, R. Sabourin, L. E. S. Oliveira. "Dynamic Selection of Classifiers — A
  Comprehensive Review." *Pattern Recognition*, vol. 47, no. 11, 2014.
