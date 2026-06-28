# Experimento multi-base — OES/DES/SES em 8 bases

Mesmo pipeline do artigo (pool da Tabela 3 → SES-GA com fitness de acurácia+diversidade → DES → OES → 5 métricas → Wilcoxon), aplicado a 8 bases em dois modos de pré-processamento. Métricas sempre no **esforço bruto** (alvo log-transformado só no treino quando assimétrico). Para SES/DES/OES menor sMAPE/MRE/MASE é melhor; maior NSE/COD é melhor.

## Bases, categoria e procedência (sem duplicatas)

| Base | Categoria | Fonte |
|---|---|---|
| finnish | SEE | TIEKE (Finlandia), 1997; repositorio PROMISE/figshare. Base do artigo. |
| maxwell | SEE | K.D. Maxwell, banco comercial finlandes; Applied Statistics for Software Managers (Prentice-Hall, 2002); Zenodo. Base do artigo. |
| desharnais | SEE | J.M. Desharnais (1989), tese de mestrado, UQAM; 81 projetos canadenses; repositorio PROMISE. |
| china | SEE | China dataset (499 projetos por pontos de funcao); repositorio PROMISE / derivado do ISBSG. |
| kitchenham | SEE | Kitchenham, Pfleeger, McColl & Eagan (2002), 'An empirical study of maintenance and development estimation accuracy'; PROMISE. |
| coc81 | SEE | B. Boehm, Software Engineering Economics (Prentice-Hall, 1981); 63 projetos COCOMO; repositorio PROMISE (coc81). |
| debutanizer | regressao generica (nao-SEE) | Coluna debutanizadora (sensor virtual de processo quimico). Fortuna et al., Soft Sensors for Monitoring and Control of Industrial Processes (Springer, 2007); benchmark do OpenML. NAO e estimacao de esforco. |
| abalone | regressao generica (nao-SEE) | Abalone (prever idade/aneis a partir de medidas fisicas). Nash et al. (1994); UCI Machine Learning Repository. NAO e estimacao de esforco. |

> **debutanizer** e **abalone** foram incluídas a pedido para avaliar os 3 métodos, mas **não são** problemas de estimação de esforço de software — são benchmarks de regressão genérica. Servem de contraste: como têm features fortes, os 3 métodos atingem sMAPE baixo (~15%) nelas, evidenciando que o sMAPE alto nas bases de SEE é **intrínseco à dificuldade de estimar esforço**, não falha do método.

## Pré-processamento identificado por base (modo melhorado)

| Base | Linhas | Features (base→melh) | log alvo | Removidos no melhorado |
|---|---|---|---|---|
| finnish | 405 | 37→37 | sim | — |
| maxwell | 62 | 24→15 | sim | MI≈0: ['Har', 'Syear', 'T01', 'T03', 'T04', 'T05', 'T06', 'T13', 'Telonuse'] |
| desharnais | 77 | 8→6 | sim | corr>0.98: ['PointsAjust']; MI≈0: ['TeamExp'] |
| china | 499 | 10→10 | sim | constantes: ['Dev.Type'] |
| kitchenham | 135 | 5→4 | sim | MI≈0: ['First.estimate.method'] |
| coc81 | 63 | 15→10 | sim | constantes: ['docu', 'flex', 'pcon', 'prec', 'resl', 'ruse', 'site', 'team']; MI≈0: ['cplx', 'ltex', 'plex', 'pvol', 'sced'] |
| debutanizer | 2394 | 7→6 | sim | corr>0.98: ['u7'] |
| abalone | 4177 | 8→7 | sim | corr>0.98: ['diameter'] |

(IDs, datas, strings e colunas de **vazamento** — derivadas do esforço ou só conhecidas ao fim — são removidas nos dois modos. Ex.: China `PDR_*`/`N_effort`; `Duration`/`Length`/`Time`; COCOMO `months`.)

## O OES (Proposto) é o melhor? (holdout, posição entre 15 modelos)

| Base | Modo | OES é o melhor em | rank mediano do OES | quem vence sMAPE |
|---|---|---|---|---|
| finnish | baseline | 1/5 metricas | 2º | ET |
| finnish | melhorado | 0/5 metricas | 10º | ET |
| maxwell | baseline | 0/5 metricas | 5º | BG |
| maxwell | melhorado | 0/5 metricas | 5º | MLP |
| desharnais | baseline | 2/5 metricas | 2º | Proposed |
| desharnais | melhorado | 0/5 metricas | 3º | DES |
| china | baseline | 0/5 metricas | 4º | DES |
| china | melhorado | 0/5 metricas | 12º | ET |
| kitchenham | baseline | 2/5 metricas | 2º | ET |
| kitchenham | melhorado | 1/5 metricas | 10º | ET |
| coc81 | baseline | 0/5 metricas | 4º | Static |
| coc81 | melhorado | 0/5 metricas | 9º | DT |
| debutanizer | baseline | 0/5 metricas | 3º | ET |
| debutanizer | melhorado | 0/5 metricas | 3º | ET |
| abalone | baseline | 1/5 metricas | 2º | Proposed |
| abalone | melhorado | 1/5 metricas | 2º | XGB |

## Teste de hipótese: de H0 para H1 (quem é quem)

O teste de Wilcoxon do artigo (Tabela 6) compara **predito × real** de cada modelo — diz se as predições diferem dos valores reais, mas não diz qual método é melhor. Para decidir **quem vence**, fazemos um teste **pareado método × método** sobre os erros absolutos por instância, em dois passos:

1. **H0**: os dois métodos têm o mesmo desempenho (mediana das diferenças de erro = 0). **H1 (bilateral)**: diferem. Calcula-se o p-valor de Wilcoxon pareado.
2. **Se p ≥ 0,05** → não rejeita H0 → *empate estatístico* (não dá para eleger vencedor). **Se p < 0,05** → rejeita H0; aí a **direção** do efeito (método com menor erro absoluto mediano, equivalente ao sinal predominante das diferenças) define a H1 específica *‘método A erra menos que B’* — é isso que responde **quem é quem**. Não basta dizer que rejeitou H0.

Para 3 métodos fazemos as 3 comparações par a par (SES×DES, SES×OES, DES×OES). Com poucas comparações, pode-se ainda aplicar correção de Holm ao risco de múltiplos testes. (Tabela no modo melhorado, conjunto de teste; em bases com teste pequeno o poder do teste é baixo — daí muitos ‘empates’.)

> **Cuidado com o tamanho amostral.** Significância ≠ relevância prática. Em bases grandes (debutanizer n≈2,4k; abalone n≈4,2k) diferenças minúsculas já rejeitam H0; em bases pequenas (COCOMO81 tem ~19 pontos de teste) mesmo diferenças reais não atingem significância. Por isso o ranking por erro mediano acompanha o p-valor na tabela.

| Base | Categoria | Ranking (menor erro →) | Par | med\|err\| A | med\|err\| B | p | Rejeita H0 | Vencedor (H1) |
|---|---|---|---|---|---|---|---|---|
| finnish | SEE | SES > OES > DES | SES vs DES | 936.8 | 1143.9 | 0.0434 | sim | SES |
| finnish | SEE | SES > OES > DES | SES vs OES | 936.8 | 1071.5 | 0.2148 | nao | empate |
| finnish | SEE | SES > OES > DES | DES vs OES | 1143.9 | 1071.5 | 0.0129 | sim | OES |
| maxwell | SEE | SES > DES > OES | SES vs DES | 1310.9 | 1610.5 | 0.5153 | nao | empate |
| maxwell | SEE | SES > DES > OES | SES vs OES | 1310.9 | 1890.6 | 0.6226 | nao | empate |
| maxwell | SEE | SES > DES > OES | DES vs OES | 1610.5 | 1890.6 | 0.3321 | nao | empate |
| desharnais | SEE | DES > OES > SES | SES vs DES | 1002.8 | 628.2 | 0.1875 | nao | empate |
| desharnais | SEE | DES > OES > SES | SES vs OES | 1002.8 | 855.0 | 0.16 | nao | empate |
| desharnais | SEE | DES > OES > SES | DES vs OES | 628.2 | 855.0 | 0.2768 | nao | empate |
| china | SEE | SES > OES > DES | SES vs DES | 863.7 | 956.4 | 0.2687 | nao | empate |
| china | SEE | SES > OES > DES | SES vs OES | 863.7 | 941.3 | 0.5716 | nao | empate |
| china | SEE | SES > OES > DES | DES vs OES | 956.4 | 941.3 | 0.0409 | sim | OES |
| kitchenham | SEE | SES > OES > DES | SES vs DES | 288.5 | 390.9 | 0.3751 | nao | empate |
| kitchenham | SEE | SES > OES > DES | SES vs OES | 288.5 | 342.6 | 0.1518 | nao | empate |
| kitchenham | SEE | SES > OES > DES | DES vs OES | 390.9 | 342.6 | 0.6535 | nao | empate |
| coc81 | SEE | SES > DES > OES | SES vs DES | 124.4 | 127.4 | 0.3321 | nao | empate |
| coc81 | SEE | SES > DES > OES | SES vs OES | 124.4 | 129.3 | 0.1688 | nao | empate |
| coc81 | SEE | SES > DES > OES | DES vs OES | 127.4 | 129.3 | 0.4413 | nao | empate |
| debutanizer | regressao generica (nao-SEE) | DES > OES > SES | SES vs DES | 0.024 | 0.022 | 0.0 | sim | DES |
| debutanizer | regressao generica (nao-SEE) | DES > OES > SES | SES vs OES | 0.024 | 0.023 | 0.0 | sim | OES |
| debutanizer | regressao generica (nao-SEE) | DES > OES > SES | DES vs OES | 0.022 | 0.023 | 0.0 | sim | DES |
| abalone | regressao generica (nao-SEE) | SES > DES > OES | SES vs DES | 1.075 | 1.092 | 0.2897 | nao | empate |
| abalone | regressao generica (nao-SEE) | SES > DES > OES | SES vs OES | 1.075 | 1.092 | 0.5031 | nao | empate |
| abalone | regressao generica (nao-SEE) | SES > DES > OES | DES vs OES | 1.092 | 1.092 | 0.0039 | sim | DES |

## Comparação por base — SES/DES/OES: baseline → melhorado (Δ)

### finnish

| Ens | sMAPE base→melh (Δ) | MRE base→melh (Δ) | MASE base→melh (Δ) | NSE base→melh (Δ) | COD base→melh (Δ) |
|---|---|---|---|---|---|
| SES | 55.0554→55.3569 (+0.3015) | 0.7339→0.7303 (-0.0035) | 0.3685→0.4757 (+0.1072) | 0.5701→0.2875 (-0.2826) | 0.678→0.4791 (-0.1989) |
| DES | 54.9829→62.9881 (+8.0052) | 0.7315→0.8111 (+0.0797) | 0.3759→0.5411 (+0.1652) | 0.566→0.0356 (-0.5304) | 0.6993→0.284 (-0.4153) |
| OES | 54.862→57.745 (+2.883) | 0.7278→0.758 (+0.0303) | 0.3688→0.4954 (+0.1265) | 0.5705→0.2626 (-0.3079) | 0.6946→0.4152 (-0.2794) |

### maxwell

| Ens | sMAPE base→melh (Δ) | MRE base→melh (Δ) | MASE base→melh (Δ) | NSE base→melh (Δ) | COD base→melh (Δ) |
|---|---|---|---|---|---|
| SES | 53.3984→35.1515 (-18.2469) | 1.1336→0.3941 (-0.7395) | 0.4147→0.5551 (+0.1404) | 0.4833→0.1989 (-0.2844) | 0.8478→0.7965 (-0.0512) |
| DES | 49.8802→43.5603 (-6.3199) | 0.678→0.4189 (-0.259) | 0.3945→0.5874 (+0.1929) | 0.4344→0.1537 (-0.2807) | 0.8514→0.6164 (-0.2351) |
| OES | 51.7048→37.5734 (-14.1314) | 0.9023→0.3933 (-0.5089) | 0.403→0.5666 (+0.1636) | 0.4623→0.2208 (-0.2415) | 0.8709→0.7181 (-0.1527) |

### desharnais

| Ens | sMAPE base→melh (Δ) | MRE base→melh (Δ) | MASE base→melh (Δ) | NSE base→melh (Δ) | COD base→melh (Δ) |
|---|---|---|---|---|---|
| SES | 39.0972→39.114 (+0.0168) | 0.509→0.4845 (-0.0245) | 0.5434→0.4861 (-0.0573) | 0.1276→0.5313 (+0.4037) | 0.2397→0.5445 (+0.3048) |
| DES | 39.4102→30.18 (-9.2303) | 0.5041→0.3309 (-0.1732) | 0.5428→0.4076 (-0.1351) | 0.1084→0.6319 (+0.5236) | 0.2373→0.6489 (+0.4116) |
| OES | 37.9041→34.9254 (-2.9786) | 0.4908→0.4063 (-0.0845) | 0.5389→0.4445 (-0.0945) | 0.1208→0.6071 (+0.4863) | 0.2393→0.6233 (+0.384) |

### china

| Ens | sMAPE base→melh (Δ) | MRE base→melh (Δ) | MASE base→melh (Δ) | NSE base→melh (Δ) | COD base→melh (Δ) |
|---|---|---|---|---|---|
| SES | 83.9211→69.6887 (-14.2324) | 3.0867→1.4481 (-1.6387) | 0.6078→1.2162 (+0.6083) | 0.4279→-34.1192 (-34.5472) | 0.4647→0.1492 (-0.3155) |
| DES | 74.2306→70.1586 (-4.0719) | 1.7834→1.1872 (-0.5962) | 0.5368→0.4786 (-0.0581) | 0.4093→0.3759 (-0.0334) | 0.4392→0.4266 (-0.0126) |
| OES | 77.5873→70.1393 (-7.448) | 2.3978→1.3078 (-1.09) | 0.5542→0.8379 (+0.2838) | 0.4361→-8.1189 (-8.555) | 0.4586→0.1789 (-0.2797) |

### kitchenham

| Ens | sMAPE base→melh (Δ) | MRE base→melh (Δ) | MASE base→melh (Δ) | NSE base→melh (Δ) | COD base→melh (Δ) |
|---|---|---|---|---|---|
| SES | 35.8875→33.8352 (-2.0523) | 0.5562→0.5619 (+0.0057) | 0.2314→3.9463 (+3.7149) | 0.8682→-113.7782 (-114.6463) | 0.9043→0.9919 (+0.0876) |
| DES | 32.6503→30.6069 (-2.0434) | 0.3999→0.2967 (-0.1032) | 0.2163→0.428 (+0.2117) | 0.8686→0.1813 (-0.6873) | 0.8854→0.3039 (-0.5815) |
| OES | 26.3515→31.2574 (+4.9058) | 0.3838→0.3986 (+0.0148) | 0.1825→1.8624 (+1.6798) | 0.9215→-23.1022 (-24.0236) | 0.9276→0.9941 (+0.0666) |

### coc81

| Ens | sMAPE base→melh (Δ) | MRE base→melh (Δ) | MASE base→melh (Δ) | NSE base→melh (Δ) | COD base→melh (Δ) |
|---|---|---|---|---|---|
| SES | 91.3836→104.8837 (+13.5001) | 3.2828→1.6128 (-1.67) | 0.4439→0.5689 (+0.125) | 0.1216→0.3477 (+0.2261) | 0.269→0.5054 (+0.2364) |
| DES | 94.2862→105.2939 (+11.0077) | 3.8148→1.0631 (-2.7517) | 0.4541→0.4858 (+0.0317) | 0.1425→0.5123 (+0.3698) | 0.2341→0.8368 (+0.6027) |
| OES | 92.4454→104.6411 (+12.1957) | 3.542→1.3264 (-2.2155) | 0.4467→0.5258 (+0.0791) | 0.133→0.437 (+0.304) | 0.2511→0.6928 (+0.4417) |

### debutanizer

| Ens | sMAPE base→melh (Δ) | MRE base→melh (Δ) | MASE base→melh (Δ) | NSE base→melh (Δ) | COD base→melh (Δ) |
|---|---|---|---|---|---|
| SES | 22.4213→18.5327 (-3.8886) | 0.2888→16540364.9791 (+16540364.6903) | 0.3414→0.2628 (-0.0786) | 0.6858→0.7826 (+0.0969) | 0.7247→0.8027 (+0.078) |
| DES | 17.9253→16.7351 (-1.1902) | 0.2173→18096142.1781 (+18096141.9608) | 0.2811→0.24 (-0.041) | 0.7481→0.8047 (+0.0566) | 0.7614→0.8173 (+0.0559) |
| OES | 20.0406→17.5232 (-2.5175) | 0.2501→17318253.577 (+17318253.3269) | 0.3075→0.2496 (-0.0579) | 0.7236→0.7972 (+0.0736) | 0.7499→0.8143 (+0.0644) |

### abalone

| Ens | sMAPE base→melh (Δ) | MRE base→melh (Δ) | MASE base→melh (Δ) | NSE base→melh (Δ) | COD base→melh (Δ) |
|---|---|---|---|---|---|
| SES | 15.0893→14.7525 (-0.3368) | 0.1545→0.1474 (-0.007) | 0.4714→0.4474 (-0.024) | 0.5213→0.547 (+0.0257) | 0.5247→0.5521 (+0.0273) |
| DES | 14.8229→14.8478 (+0.0249) | 0.1521→0.1491 (-0.0029) | 0.4582→0.4476 (-0.0105) | 0.5457→0.5505 (+0.0048) | 0.5478→0.5554 (+0.0076) |
| OES | 14.7977→14.6695 (-0.1282) | 0.1515→0.1469 (-0.0046) | 0.4594→0.4436 (-0.0158) | 0.5433→0.5557 (+0.0125) | 0.5449→0.5607 (+0.0158) |

> A robustez (mediana ± dp em CV repetida) é acrescentada por `python src/run_experiment_multi.py robustez` — recomendada porque o NSE/COD no esforço bruto é instável em bases de cauda pesada (um holdout único pode dar NSE muito negativo).

## Robustez — CV repetida (mediana ± desvio), OES baseline vs melhorado

Mediana e desvio das métricas do OES sobre 5 splits estratificados. A **mediana** é robusta a partições catastróficas (NSE muito negativo num split isolado). Bases grandes (debutanizer, abalone) foram subamostradas (cap=700) só aqui, para a robustez rodar em tempo viável.

| Base | Modo | sMAPE (med±dp) | MRE (med±dp) | MASE (med±dp) | NSE (med±dp) | COD (med±dp) |
|---|---|---|---|---|---|---|
| finnish | baseline | 62.494 ± 3.274 | 1.045 ± 0.142 | 0.429 ± 0.037 | 0.556 ± 0.044 | 0.588 ± 0.039 |
| finnish | melhorado | 56.871 ± 3.449 | 0.809 ± 0.118 | 0.467 ± 0.084 | 0.5 ± 0.837 | 0.528 ± 0.15 |
| maxwell | baseline | 60.026 ± 6.401 | 1.017 ± 0.279 | 0.417 ± 0.258 | 0.514 ± 0.715 | 0.587 ± 0.183 |
| maxwell | melhorado | 48.532 ± 7.458 | 0.752 ± 0.967 | 0.448 ± 0.057 | 0.662 ± 0.205 | 0.728 ± 0.232 |
| desharnais | baseline | 40.287 ± 8.172 | 0.551 ± 0.209 | 0.5 ± 0.066 | 0.453 ± 0.128 | 0.474 ± 0.087 |
| desharnais | melhorado | 34.53 ± 3.966 | 0.429 ± 0.1 | 0.466 ± 0.033 | 0.446 ± 0.136 | 0.461 ± 0.156 |
| china | baseline | 69.869 ± 3.409 | 1.385 ± 0.516 | 0.525 ± 0.027 | 0.395 ± 0.129 | 0.406 ± 0.104 |
| china | melhorado | 68.553 ± 3.784 | 1.033 ± 0.22 | 0.66 ± 0.294 | -2.445 ± 21.501 | 0.326 ± 0.132 |
| kitchenham | baseline | 37.609 ± 9.328 | 0.562 ± 0.256 | 0.283 ± 0.074 | 0.758 ± 0.114 | 0.847 ± 0.055 |
| kitchenham | melhorado | 29.615 ± 3.519 | 0.321 ± 0.066 | 0.299 ± 0.661 | 0.62 ± 9.039 | 0.863 ± 0.097 |
| coc81 | baseline | 121.156 ± 18.952 | 8.478 ± 5.176 | 0.589 ± 0.331 | 0.043 ± 2.555 | 0.273 ± 0.17 |
| coc81 | melhorado | 94.789 ± 11.175 | 3.234 ± 2.094 | 0.445 ± 0.127 | 0.41 ± 0.469 | 0.836 ± 0.334 |
| debutanizer | baseline | 28.872 ± 1.898 | 0.413 ± 59368124.033 | 0.423 ± 0.022 | 0.611 ± 0.017 | 0.629 ± 0.019 |
| debutanizer | melhorado | 27.493 ± 1.48 | 0.394 ± 0.058 | 0.426 ± 0.031 | 0.615 ± 0.057 | 0.636 ± 0.066 |
| abalone | baseline | 15.229 ± 0.881 | 0.147 ± 0.01 | 0.463 ± 0.047 | 0.484 ± 0.06 | 0.518 ± 0.051 |
| abalone | melhorado | 15.272 ± 0.955 | 0.148 ± 0.01 | 0.452 ± 0.038 | 0.508 ± 0.055 | 0.521 ± 0.045 |
