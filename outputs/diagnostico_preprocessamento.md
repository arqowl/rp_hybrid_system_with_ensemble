# Diagnóstico de Pré-processamento — Finnish e Maxwell

Avaliação das duas bases **com o mesmo recorte de features do pipeline** (antes da normalização). Não altera a replicação; serve para decidir o que faz sentido melhorar. Figuras em `outputs/figures/diagnostico/`.

## Finnish  (405 amostras × 30 features, alvo incl. à parte)

**Distribuição do alvo (desbalanceamento em regressão)**

- Skewness cru = **3.69**, kurtose (excess) = **18.45** → fortemente assimétrico à direita. Após `log1p`: skew = **-0.13**, kurtose = **-0.27** (bem mais próximo do normal).
- Outliers do alvo (IQR): **35** de 405. Máx/mediana = **25.5×**.

**Ausentes / sentinelas**

- Células ausentes (NaN) no X selecionado: **0**. Colunas que contêm algum zero: 8.

**Variância quase-nula**

- Features constantes: nenhuma.
- Variância quase-nula (<0.01 após MinMax): ['AlgFP', 'IntFP'].

**Redundância (correlação alta entre features)**

- Pares com |r| > 0.95: **0**.
- Pares com |r| > 0.90 (1): [('Size_ep99_proj', 'AllFP_ep20', np.float64(0.913))]

**Multicolinearidade**

- Features com VIF > 10: **8** → [('SituCoeff', 10.6), ('InpFP', 'inf'), ('InqFP', 'inf'), ('OutFP', 'inf'), ('IntFP', 'inf'), ('EntFP', 'inf'), ('AlgFP', 'inf'), ('AllFP_ep20', 'inf')]
- Número de condição da matriz (padronizada): **9** (ok).

**Relevância feature→alvo (informação mútua)**

- Top-5 informativas: [('Size_ep99_proj', 0.577), ('AllFP_ep20', 0.429), ('SituCoeff', 0.328), ('EntFP', 0.314), ('InpFP', 0.198)].
- Sem informação (MI≈0): nenhuma.

**Dimensionalidade**

- Razão amostras/features = **13.5**.
- PCA: **20** componentes para 90% da variância, **23** para 95% (de 30 features).

## Maxwell  (62 amostras × 20 features, alvo incl. à parte)

**Distribuição do alvo (desbalanceamento em regressão)**

- Skewness cru = **3.27**, kurtose (excess) = **12.52** → fortemente assimétrico à direita. Após `log1p`: skew = **0.06**, kurtose = **-0.30** (bem mais próximo do normal).
- Outliers do alvo (IQR): **5** de 62. Máx/mediana = **12.3×**.

**Ausentes / sentinelas**

- Células ausentes (NaN) no X selecionado: **0**. Colunas que contêm algum zero: 1.

**Variância quase-nula**

- Features constantes: nenhuma.
- Variância quase-nula (<0.01 após MinMax): nenhuma.

**Redundância (correlação alta entre features)**

- Pares com |r| > 0.95: **0**.
- Pares com |r| > 0.90 (0): []

**Multicolinearidade**

- Features com VIF > 10: **0** → []
- Número de condição da matriz (padronizada): **6** (ok).

**Relevância feature→alvo (informação mútua)**

- Top-5 informativas: [('Size', 0.443), ('T07', 0.169), ('T11', 0.092), ('T09', 0.081), ('App', 0.046)].
- Sem informação (MI≈0): ['Har', 'T01', 'T04', 'T06', 'T13'].

**Dimensionalidade**

- Razão amostras/features = **3.1**.
- PCA: **13** componentes para 90% da variância, **16** para 95% (de 20 features).
