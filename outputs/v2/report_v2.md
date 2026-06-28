# Relatório v2 — Pré-processamento melhorado (comparação com a replicação fiel)

Esta variante mantém **a mesma arquitetura** (pool da Tabela 3 → SES-GA com
fitness de acurácia+diversidade → DES → OES → 5 métricas → Wilcoxon) e muda
**apenas o pré-processamento**, para isolar o efeito dos dados. Todas as
métricas seguem calculadas no **esforço bruto** (o alvo é log-transformado só
para o treino e revertido na avaliação), então os números são comparáveis com a
execução anterior (colunas `v1`).

## O que mudou no pré-processamento (3 melhorias)

1. **Alvo em log1p (+MinMax)** nas duas bases — corrige a forte assimetria
   (skew 3.7/3.3 → ~0). Treina-se no espaço log; as métricas são revertidas
   para o esforço bruto.
2. **Remoção direcionada de atributos** (além do recorte da replicação):
   - Finnish (removidos a mais): `AllFP_ep20` (agregado = soma das componentes,
     VIF = ∞), `IntFP`, `AlgFP` (variância quase-nula) → conjunto final:
     28 features (incl. alvo).
   - Maxwell (removidos a mais): `Har`, `T01`, `T04`, `T06`, `T13` (informação
     mútua ≈ 0) → conjunto final: 16 features
     (incl. alvo).
3. **Split 70:30 estratificado por quantis do alvo** (em vez de aleatório puro)
   + **validação cruzada repetida** (10 splits) para estimativas estáveis.

> Não foram aplicados **PCA** nem **resampling de regressão**: o diagnóstico
> mostrou baixa compressibilidade (95% da variância exige 23/30 e 16/20
> componentes), tornando-os de baixo valor aqui. Podem ser adicionados se
> desejado.

## Tabela 4 — Finnish: ensembles v2 × v1 × artigo

| Modelo | sMAPE v2 | sMAPE v1 | Δ | art | MRE v2 | MRE v1 | Δ | art | MASE v2 | MASE v1 | Δ | art | NSE v2 | NSE v1 | Δ | art | COD v2 | COD v1 | Δ | art |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| SES | 54.1895 | 66.487 | -12.2975 | 27.825 | 0.6622 | 1.4043 | -0.7421 | 1.05944 | 0.4356 | 0.4295 | 0.0061 | 0.54115 | 0.4902 | 0.5512 | -0.061 | 0.71622 | 0.565 | 0.6623 | -0.0973 | 0.87652 |
| DES | 58.6333 | 68.2034 | -9.5701 | 33.884 | 0.7007 | 1.3231 | -0.6224 | 0.92209 | 0.4785 | 0.4263 | 0.0522 | 0.5723 | 0.3829 | 0.5473 | -0.1644 | 0.66953 | 0.4513 | 0.6614 | -0.2101 | 0.84379 |
| OES | 55.4505 | 64.8937 | -9.4432 | 23.896 | 0.6725 | 1.313 | -0.6405 | 0.79142 | 0.4536 | 0.4246 | 0.029 | 0.45167 | 0.4641 | 0.5516 | -0.0875 | 0.78121 | 0.5232 | 0.6678 | -0.1446 | 0.91375 |

(Comparação completa de todos os modelos em `outputs/v2/tables/tabela4_finnish_v2.csv`.)

## Tabela 5 — Maxwell: ensembles v2 × v1 × artigo

| Modelo | sMAPE v2 | sMAPE v1 | Δ | art | MRE v2 | MRE v1 | Δ | art | MASE v2 | MASE v1 | Δ | art | NSE v2 | NSE v1 | Δ | art | COD v2 | COD v1 | Δ | art |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| SES | 43.8123 | 60.2988 | -16.4865 | 23.527 | 0.555 | 1.0315 | -0.4765 | 0.7629 | 0.3133 | 0.4317 | -0.1184 | 0.42057 | 0.8586 | 0.4559 | 0.4027 | 0.8042 | 0.8943 | 0.8589 | 0.0354 | 0.93988 |
| DES | 42.2781 | 55.2651 | -12.987 | 16.558 | 0.5353 | 0.6099 | -0.0746 | 0.50287 | 0.2636 | 0.4414 | -0.1778 | 0.32678 | 0.9301 | 0.3029 | 0.6272 | 0.90453 | 0.9364 | 0.6592 | 0.2772 | 0.9522 |
| OES | 42.4013 | 59.1452 | -16.7439 | 15.057 | 0.5383 | 0.8148 | -0.2765 | 0.47942 | 0.2777 | 0.4359 | -0.1582 | 0.26125 | 0.9011 | 0.3852 | 0.5159 | 0.95146 | 0.9198 | 0.8019 | 0.1179 | 0.98359 |

(Comparação completa em `outputs/v2/tables/tabela5_maxwell_v2.csv`.)

## Tabela 6 — Wilcoxon (p-value): v2 × v1

| Modelo | Finnish v2 | Finnish v1 | Maxwell v2 | Maxwell v1 |
|---|---|---|---|---|
| Static Ensemble Selection (SES) | 0.0895 | 0.0015 | 0.9843 | 0.3525 |
| Dynamic Ensemble Selection (DES) | 0.0588 | 0.032 | 0.5678 | 0.2753 |
| Omni Ensemble Selection (OES) | 0.0739 | 0.006 | 0.768 | 0.3736 |

## Robustez — validação cruzada repetida (10 splits estratificados)

Média ± desvio-padrão das métricas dos 3 ensembles (esforço bruto). O desvio
indica o quanto o resultado de um único holdout é confiável — relevante
sobretudo no Maxwell (teste ~19 pontos).

| Dataset | Ensemble | sMAPE (média±dp) | MRE (média±dp) | MASE (média±dp) | NSE (média±dp) | COD (média±dp) |
|---|---|---|---|---|---|---|
| Finnish | SES | 56.5608 ± 4.7006 | 0.96 ± 0.2452 | 0.4923 ± 0.1244 | 0.0604 ± 1.1354 | 0.5509 ± 0.1369 |
| Finnish | DES | 55.979 ± 3.9244 | 0.8263 ± 0.1275 | 0.4856 ± 0.0687 | 0.2718 ± 0.3662 | 0.4062 ± 0.1178 |
| Finnish | OES | 55.4958 ± 4.0521 | 0.8794 ± 0.16 | 0.4708 ± 0.0778 | 0.3386 ± 0.4292 | 0.5272 ± 0.1228 |
| Maxwell | SES | 47.9048 ± 7.1162 | 4.3407 ± 7.7513 | 0.9166 ± 1.0738 | -4.7466 ± 13.0248 | 0.5924 ± 0.3762 |
| Maxwell | DES | 46.4441 ± 10.4777 | 0.654 ± 0.2455 | 0.5268 ± 0.4038 | -0.2453 ± 2.798 | 0.7672 ± 0.2343 |
| Maxwell | OES | 46.1655 ± 8.8728 | 2.4759 ± 3.8896 | 0.7016 ± 0.6443 | -1.2128 ± 3.7275 | 0.6404 ± 0.3538 |

## Como ler a comparação

- **NSE/COD**: maior é melhor. **sMAPE/MRE/MASE**: menor é melhor.
- A coluna **Δ** é (v2 − v1) no esforço bruto. Δ negativo em sMAPE/MRE/MASE e
  positivo em NSE/COD indicam **melhora** com o pré-processamento.
- A coluna **art** repete o valor do artigo (referência), lembrando que a
  igualdade decimal não é esperada (features exatas e hiperparâmetros do GA/DES
  não constam do artigo).

Arquivos: tabelas em `outputs/v2/tables/`, figuras em `outputs/v2/figures/`.
