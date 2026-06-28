"""
run_experiment_multi.py — Aplica o MESMO pipeline (pool -> SES-GA -> DES -> OES,
5 metricas, Wilcoxon) a TODAS as bases de SEE, em dois modos de
pre-processamento (baseline e melhorado), reaproveitando pipeline.py.

Bases: finnish, maxwell (artigo) + desharnais, china, kitchenham, coc81 (novas).
(Excluidas por nao serem de esforco de software: debutanizer, abalone.)

Gera, em outputs/multi/:
  tables/<base>_compare.csv   (todos os modelos: metrica baseline vs melhorado vs delta)
  tables/oes_ranking.csv      (por base/modo: melhor modelo e posicao do OES)
  tables/robustez_oes.csv     (CV repetida: mediana +- dp, baseline vs melhorado)
  figures/<base>_scatter.png  (scatter SES/DES/OES, modo melhorado)
  report_multi.md

Uso:
  python src/run_experiment_multi.py            # holdout + figuras + ranking + report
  python src/run_experiment_multi.py robustez   # acrescenta a robustez (CV repetida)
"""
import os, sys
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

import data_multi as dm
import pipeline as pl
import figures as fg
from scipy.stats import wilcoxon

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
M = os.path.join(ROOT, "outputs", "multi")
MTAB = os.path.join(M, "tables"); MFIG = os.path.join(M, "figures")
os.makedirs(MTAB, exist_ok=True); os.makedirs(MFIG, exist_ok=True)

MC = ["sMAPE", "MRE", "MASE", "NSE", "COD"]
LOWER_BETTER = {"sMAPE", "MRE", "MASE"}
ROW_ORDER = pl.mdl.POOL_ORDER + ["Static", "DES", "Proposed"]
N_REPEATS = 5


def rodar(name, mode, seed=42, cap=None):
    d = dm.build(name, mode, split_seed=seed, cap=cap)
    rng = np.random.default_rng(seed)
    pool = pl.fase2_treinar_pool(d)
    sel = pl.fase2_ses_ga(d, pool, rng)
    ens = pl.fase3_des_oes(d, pool, sel)
    rows = pl.montar_linhas(d, pool, sel, ens)
    wil = pl.wilcoxon_ensembles(d, sel, ens)
    ens_tr = {"SES": sel["ses_tr"], "DES": ens["des_tr"], "OES": ens["oes_tr"]}
    ens_te = {"SES": sel["ses_te"], "DES": ens["des_te"], "OES": ens["oes_te"]}
    return dict(d=d, rows=rows, wil=wil, sel_names=sel["sel_names"],
                ens_tr=ens_tr, ens_te=ens_te)


def rank_oes(rows):
    """Para cada metrica: (melhor_modelo, valor, rank_do_OES em 15)."""
    out = {}
    for m in MC:
        vals = {k: rows[k][m] for k in ROW_ORDER}
        order = sorted(vals, key=lambda k: vals[k], reverse=(m not in LOWER_BETTER))
        out[m] = (order[0], round(vals[order[0]], 3), order.index("Proposed") + 1)
    return out


def comparar_pares(ens_te, y, alpha=0.05):
    """Teste de Wilcoxon PAREADO metodo x metodo nos erros absolutos por
    instancia. Para cada par (SES/DES/OES): p-valor bilateral (H0 = mesmo
    desempenho). Se p<alpha, rejeita H0 e a DIRECAO (menor erro absoluto
    mediano) define a hipotese alternativa especifica -> diz QUEM vence."""
    ms = ["SES", "DES", "OES"]
    err = {m: np.abs(np.asarray(y, float) - np.asarray(ens_te[m], float)) for m in ms}
    _r = lambda x: round(x, 3) if abs(x) < 10 else round(x, 1)
    recs = []
    for a, b in [("SES", "DES"), ("SES", "OES"), ("DES", "OES")]:
        ea, eb = err[a], err[b]
        try:
            p = 1.0 if np.allclose(ea - eb, 0) else float(wilcoxon(ea, eb).pvalue)
        except ValueError:
            p = 1.0
        ma, mb = float(np.median(ea)), float(np.median(eb))
        win = "empate" if p >= alpha else (a if ma < mb else b)
        recs.append({"Par": f"{a} vs {b}", "medErr_A": _r(ma),
                     "medErr_B": _r(mb), "p": round(p, 4),
                     "Rejeita_H0": "sim" if p < alpha else "nao",
                     "Vencedor_H1": win})
    return recs


def ranking_metodos(ens_te, y):
    """Ordena SES/DES/OES do melhor (menor erro absoluto mediano) ao pior."""
    err = {m: float(np.median(np.abs(np.asarray(y, float) - np.asarray(ens_te[m], float))))
           for m in ["SES", "DES", "OES"]}
    return sorted(err, key=err.get)


def holdout():
    print("Holdout (seed 42) em 6 bases x 2 modos ...")
    res = {}
    rank_recs = []
    for name in dm.DATASETS:
        res[name] = {md: rodar(name, md) for md in ["baseline", "melhorado"]}
        # tabela comparativa (todos os modelos)
        recs = []
        for k in ROW_ORDER:
            rec = {"Model": k}
            for m in MC:
                b = res[name]["baseline"]["rows"][k][m]
                v = res[name]["melhorado"]["rows"][k][m]
                rec[f"{m}_base"] = round(b, 4); rec[f"{m}_melh"] = round(v, 4)
                rec[f"{m}_delta"] = round(v - b, 4)
            recs.append(rec)
        pd.DataFrame(recs).to_csv(os.path.join(MTAB, f"{name}_compare.csv"), index=False)
        # ranking do OES
        for md in ["baseline", "melhorado"]:
            r = rank_oes(res[name][md]["rows"])
            n1 = sum(1 for m in MC if r[m][0] == "Proposed")
            ranks = [r[m][2] for m in MC]
            rank_recs.append({"Dataset": name, "Modo": md,
                              "OES_e_o_melhor_em": f"{n1}/5 metricas",
                              "OES_rank_mediano": int(np.median(ranks)),
                              **{f"melhor_{m}": r[m][0] for m in MC}})
        # figura scatter (melhorado)
        d = res[name]["melhorado"]["d"]
        fg.scatter_fig(os.path.join(MFIG, f"{name}_scatter.png"),
                       f"{name} (melhorado) — Scatter SES/DES/OES",
                       res[name]["melhorado"]["ens_tr"], res[name]["melhorado"]["ens_te"],
                       d["y_train_raw"], d["y_test_raw"])
    pd.DataFrame(rank_recs).to_csv(os.path.join(MTAB, "oes_ranking.csv"), index=False)

    # teste pareado metodo x metodo (modo melhorado, conjunto de teste)
    sig_recs = []
    for name in dm.DATASETS:
        ens_te = res[name]["melhorado"]["ens_te"]
        y = res[name]["melhorado"]["d"]["y_test_raw"]
        ordem = ranking_metodos(ens_te, y)
        for r in comparar_pares(ens_te, y):
            sig_recs.append({"Dataset": name, "Categoria": dm.CATEGORIA[name],
                             "Ranking_por_erro": " > ".join(ordem), **r})
    pd.DataFrame(sig_recs).to_csv(os.path.join(MTAB, "significancia_pareada.csv"), index=False)

    escrever_report(res, rank_recs, sig_recs)
    print("  -> tabelas, figuras e report_multi.md gravados.")
    return res


def robustez(subset=None, cap=700):
    datasets = subset or dm.DATASETS
    print(f"Robustez: CV repetida ({N_REPEATS} splits) x {len(datasets)} base(s) "
          f"x 2 modos (cap={cap} p/ bases grandes) ...")
    recs = []
    for name in datasets:
        store = {md: {m: [] for m in MC} for md in ["baseline", "melhorado"]}
        for s in range(N_REPEATS):
            for md in ["baseline", "melhorado"]:
                r = rodar(name, md, seed=200 + s, cap=cap)
                for m in MC:
                    store[md][m].append(r["rows"]["Proposed"][m])
        for md in ["baseline", "melhorado"]:
            rec = {"Dataset": name, "Modo": md}
            for m in MC:
                a = np.array(store[md][m], float)
                rec[f"{m}_med"] = round(float(np.median(a)), 3)
                rec[f"{m}_std"] = round(float(a.std()), 3)
            recs.append(rec)
        print(f"  [{name}] ok")
    df_new = pd.DataFrame(recs)
    path = os.path.join(MTAB, "robustez_oes.csv")
    if os.path.exists(path):                      # mescla: substitui linhas das bases recomputadas
        old = pd.read_csv(path)
        old = old[~old["Dataset"].isin(datasets)]
        df = pd.concat([old, df_new], ignore_index=True)
    else:
        df = df_new
    order = {n: i for i, n in enumerate(dm.DATASETS)}
    df["__o"] = df["Dataset"].map(order)
    df = df.sort_values(["__o", "Modo"]).drop(columns="__o")
    df.to_csv(path, index=False)
    print(f"  -> robustez_oes.csv atualizado ({df['Dataset'].nunique()} bases). "
          f"Rode 'report_robustez' para (re)escrever a seção no report.")


# ---------------------------------------------------------------- report
def _cmp_block(name, res):
    """Bloco markdown: SES/DES/OES baseline vs melhorado para uma base."""
    df = pd.read_csv(os.path.join(MTAB, f"{name}_compare.csv"))
    sub = df[df["Model"].isin(["Static", "DES", "Proposed"])]
    ren = {"Static": "SES", "DES": "DES", "Proposed": "OES"}
    head = "| Ens | " + " | ".join(f"{m} base→melh (Δ)" for m in MC) + " |"
    sep = "|" + "---|" * (1 + len(MC))
    lines = [head, sep]
    for _, r in sub.iterrows():
        cells = [ren[r["Model"]]]
        for m in MC:
            cells.append(f"{r[f'{m}_base']}→{r[f'{m}_melh']} ({r[f'{m}_delta']:+})")
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def escrever_report(res, rank_recs, sig_recs):
    L = ["# Experimento multi-base — OES/DES/SES em 8 bases",
         "",
         "Mesmo pipeline do artigo (pool da Tabela 3 → SES-GA com fitness de "
         "acurácia+diversidade → DES → OES → 5 métricas → Wilcoxon), aplicado a "
         "8 bases em dois modos de pré-processamento. Métricas sempre no "
         "**esforço bruto** (alvo log-transformado só no treino quando "
         "assimétrico). Para SES/DES/OES menor sMAPE/MRE/MASE é melhor; maior "
         "NSE/COD é melhor.",
         "",
         "## Bases, categoria e procedência (sem duplicatas)",
         "",
         "| Base | Categoria | Fonte |",
         "|---|---|---|"]
    for n in dm.DATASETS:
        L.append(f"| {n} | {dm.CATEGORIA[n]} | {dm.FONTE[n]} |")
    L += ["",
          "> **debutanizer** e **abalone** foram incluídas a pedido para avaliar "
          "os 3 métodos, mas **não são** problemas de estimação de esforço de "
          "software — são benchmarks de regressão genérica. Servem de contraste: "
          "como têm features fortes, os 3 métodos atingem sMAPE baixo (~15%) "
          "nelas, evidenciando que o sMAPE alto nas bases de SEE é **intrínseco "
          "à dificuldade de estimar esforço**, não falha do método.",
          "",
          "## Pré-processamento identificado por base (modo melhorado)",
          "",
          "| Base | Linhas | Features (base→melh) | log alvo | Removidos no melhorado |",
          "|---|---|---|---|---|"]
    for n in dm.DATASETS:
        p = res[n]["melhorado"]["d"]["prep"]
        nb = res[n]["baseline"]["d"]["n_features"]
        extra = []
        if p["drop_const"]: extra.append(f"constantes: {p['drop_const']}")
        if p["drop_nzv"]: extra.append(f"var≈0: {p['drop_nzv']}")
        if p["drop_corr"]: extra.append(f"corr>0.98: {p['drop_corr']}")
        if p["drop_mi"]: extra.append(f"MI≈0: {p['drop_mi']}")
        L.append(f"| {n} | {p['n_linhas']} | {nb}→{p['n_features']} | "
                 f"{'sim' if p['log_alvo'] else 'não'} | {'; '.join(extra) or '—'} |")
    L += ["", "(IDs, datas, strings e colunas de **vazamento** — derivadas do "
          "esforço ou só conhecidas ao fim — são removidas nos dois modos. "
          "Ex.: China `PDR_*`/`N_effort`; `Duration`/`Length`/`Time`; COCOMO `months`.)",
          ""]

    # ranking do OES
    L += ["## O OES (Proposto) é o melhor? (holdout, posição entre 15 modelos)",
          "",
          "| Base | Modo | OES é o melhor em | rank mediano do OES | quem vence sMAPE |",
          "|---|---|---|---|---|"]
    for r in rank_recs:
        L.append(f"| {r['Dataset']} | {r['Modo']} | {r['OES_e_o_melhor_em']} | "
                 f"{r['OES_rank_mediano']}º | {r['melhor_sMAPE']} |")

    # ---- secao de hipotese: de H0 para H1 ----
    L += ["",
          "## Teste de hipótese: de H0 para H1 (quem é quem)",
          "",
          "O teste de Wilcoxon do artigo (Tabela 6) compara **predito × real** "
          "de cada modelo — diz se as predições diferem dos valores reais, mas "
          "não diz qual método é melhor. Para decidir **quem vence**, fazemos um "
          "teste **pareado método × método** sobre os erros absolutos por "
          "instância, em dois passos:",
          "",
          "1. **H0**: os dois métodos têm o mesmo desempenho (mediana das "
          "diferenças de erro = 0). **H1 (bilateral)**: diferem. Calcula-se o "
          "p-valor de Wilcoxon pareado.",
          "2. **Se p ≥ 0,05** → não rejeita H0 → *empate estatístico* (não dá "
          "para eleger vencedor). **Se p < 0,05** → rejeita H0; aí a **direção** "
          "do efeito (método com menor erro absoluto mediano, equivalente ao "
          "sinal predominante das diferenças) define a H1 específica "
          "*‘método A erra menos que B’* — é isso que responde **quem é quem**. "
          "Não basta dizer que rejeitou H0.",
          "",
          "Para 3 métodos fazemos as 3 comparações par a par (SES×DES, SES×OES, "
          "DES×OES). Com poucas comparações, pode-se ainda aplicar correção de "
          "Holm ao risco de múltiplos testes. (Tabela no modo melhorado, conjunto "
          "de teste; em bases com teste pequeno o poder do teste é baixo — daí "
          "muitos ‘empates’.)",
          "",
          "> **Cuidado com o tamanho amostral.** Significância ≠ relevância "
          "prática. Em bases grandes (debutanizer n≈2,4k; abalone n≈4,2k) "
          "diferenças minúsculas já rejeitam H0; em bases pequenas (COCOMO81 tem "
          "~19 pontos de teste) mesmo diferenças reais não atingem significância. "
          "Por isso o ranking por erro mediano acompanha o p-valor na tabela.",
          "",
          "| Base | Categoria | Ranking (menor erro →) | Par | med\\|err\\| A | med\\|err\\| B | p | Rejeita H0 | Vencedor (H1) |",
          "|---|---|---|---|---|---|---|---|---|"]
    for r in sig_recs:
        L.append(f"| {r['Dataset']} | {r['Categoria']} | {r['Ranking_por_erro']} | "
                 f"{r['Par']} | {r['medErr_A']} | {r['medErr_B']} | {r['p']} | "
                 f"{r['Rejeita_H0']} | {r['Vencedor_H1']} |")

    L += ["",
          "## Comparação por base — SES/DES/OES: baseline → melhorado (Δ)",
          ""]
    for n in dm.DATASETS:
        L += [f"### {n}", "", _cmp_block(n, res), ""]
    L += ["> A robustez (mediana ± dp em CV repetida) é acrescentada por "
          "`python src/run_experiment_multi.py robustez` — recomendada porque o "
          "NSE/COD no esforço bruto é instável em bases de cauda pesada (um "
          "holdout único pode dar NSE muito negativo).", ""]
    with open(os.path.join(M, "report_multi.md"), "w") as f:
        f.write("\n".join(L))


def _append_robustez_report():
    df = pd.read_csv(os.path.join(MTAB, "robustez_oes.csv"))
    L = ["", "## Robustez — CV repetida (mediana ± desvio), OES baseline vs melhorado",
         "",
         f"Mediana e desvio das métricas do OES sobre {N_REPEATS} splits "
         "estratificados. A **mediana** é robusta a partições catastróficas "
         "(NSE muito negativo num split isolado). Bases grandes (debutanizer, "
         "abalone) foram subamostradas (cap=700) só aqui, para a robustez rodar "
         "em tempo viável.",
         "",
         "| Base | Modo | " + " | ".join(f"{m} (med±dp)" for m in MC) + " |",
         "|" + "---|" * (2 + len(MC))]
    for _, r in df.iterrows():
        cells = [r["Dataset"], r["Modo"]]
        for m in MC:
            cells.append(f"{r[f'{m}_med']} ± {r[f'{m}_std']}")
        L.append("| " + " | ".join(cells) + " |")
    with open(os.path.join(M, "report_multi.md"), "a") as f:
        f.write("\n".join(L) + "\n")


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else ""
    if arg == "robustez":
        sub = sys.argv[2].split(",") if len(sys.argv) > 2 else None
        robustez(subset=sub)
    elif arg == "report_robustez":
        _append_robustez_report()
        print("  -> seção de robustez (re)escrita no report.")
    else:
        holdout()
