"""
Pacote `src` — Sistema Híbrido de Omni-Ensemble para Estimação de Esforço (SEE).

Módulos:
    dataset_loader     T1  carregamento real + pré-processamento (5 bases)
    train_pool         T2  pool de modelos base + matrizes gabarito (in-sample e OOF)
    ses_ga_single      T3  seleção estática via GA mono-objetivo (R²)
    ses_ga_multi       T4  seleção estática via NSGA-II (R² × parcimônia)
    des_dynamic        T5  seleção dinâmica por competência local (k-NN)
    evaluator          T6  métricas (sMAPE, MRE, COD, ...)
    statistical_tests  T7  Friedman + post-hoc Bonferroni-Dunn (Demšar, 2006)
    combination            agregação das predições selecionadas
    make_results           gera tabelas e figuras finais em results/

A presença deste arquivo torna `src/` um pacote, habilitando
`from src.<modulo> import ...` no notebook e entre os módulos.
"""
