"""
Gera os resultados finais do experimento de forma reprodutível:
    results/tabela_metricas.csv          — média por método (sMAPE, MRE, COD)
    results/tabela_metricas_por_base.csv — detalhe por base
    results/teste_friedman.txt           — saída numérica Friedman + Bonferroni-Dunn
    results/figuras/*.png                — diagrama CD, ranks, Pareto, sMAPE por base

Config: pool de 5 modelos (LR, SVR, MLP, kNN, DT), seleção OUT-OF-FOLD,
alvo conforme dataset_loader.LOG_TARGET, métricas do artigo na escala original.
"""
import os, csv
import itertools
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

from src.dataset_loader import (run_pipeline, datasets_with_pool, load_pool_arrays,
                                 load_xy, load_oof_train, inverse_target)
from src.train_pool import run_train_pool, build_pool
from src.ses_ga_single import run_ses_ga_pipeline, pearson_r_squared
from src.ses_ga_multi import run_ses_ga_multi_pipeline
from src.des_dynamic import DynamicEnsembleSelector
from src.combination import combine_mean
from src.evaluator import smape, mre, cod

RESULTS = "results"; FIGS = os.path.join(RESULTS, "figuras")
CONTROL = "SES-GA-Multi"


def _predictions():
    run_pipeline(verbose=False); pool = run_train_pool(verbose=False)
    t3 = run_ses_ga_pipeline(verbose=False); t4 = run_ses_ga_multi_pipeline(verbose=False)
    names = list(build_pool().keys())
    methods = names + ["Static", "SES-GA", "SES-GA-Multi", "DES"]
    bases = datasets_with_pool()
    per_base = {}   # base -> method -> {metric: val}
    smape_mat = []
    for d in bases:
        _, pred_test, y_train, y_test = load_pool_arrays(d)
        X_train, X_test, _, _ = load_xy(d)
        oof = load_oof_train(d)
        des = DynamicEnsembleSelector(k=7, threshold=0.3); des.fit(X_train, oof, y_train)
        preds = [pred_test[:, i] for i in range(len(names))] + [
            combine_mean(pred_test, None),
            combine_mean(pred_test, t3[d]["best_chromosome"]),
            combine_mean(pred_test, t4[d]["best_balanced"]["chromosome"]),
            des.predict(X_test, pred_test),
        ]
        yte_raw = inverse_target(d, y_test)
        row_metrics = {}
        srow = []
        for m, p in zip(methods, preds):
            praw = inverse_target(d, p)
            row_metrics[m] = {"sMAPE": smape(yte_raw, praw),
                              "MRE": mre(yte_raw, praw),
                              "COD": cod(yte_raw, praw)}
            srow.append(row_metrics[m]["sMAPE"])
        per_base[d] = row_metrics
        smape_mat.append(srow)
    return methods, bases, per_base, np.array(smape_mat), t4


def _friedman(smape_mat, methods):
    K, N = smape_mat.shape
    ranks = np.zeros_like(smape_mat)
    for i in range(K):
        o = np.argsort(smape_mat[i]); rr = np.empty(N); rr[o] = np.arange(1, N + 1); ranks[i] = rr
    avg = ranks.mean(0)
    F = (12 * K / (N * (N + 1))) * (np.sum(avg ** 2) - N * (N + 1) ** 2 / 4)
    p = 1 - stats.chi2.cdf(F, N - 1)
    cd = stats.norm.ppf(1 - 0.05 / (2 * (N - 1))) * np.sqrt(N * (N + 1) / (6 * K))
    return avg, F, p, cd, K, N


def generate():
    os.makedirs(FIGS, exist_ok=True)
    methods, bases, per_base, smape_mat, t4 = _predictions()
    avg, F, p, cd, K, N = _friedman(smape_mat, methods)

    # ---- tabelas ----
    with open(os.path.join(RESULTS, "tabela_metricas_por_base.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["base", "metodo", "sMAPE", "MRE", "COD"])
        for d in bases:
            for m in methods:
                v = per_base[d][m]; w.writerow([d, m, f"{v['sMAPE']:.3f}", f"{v['MRE']:.3f}", f"{v['COD']:.3f}"])
    with open(os.path.join(RESULTS, "tabela_metricas.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["metodo", "sMAPE_medio", "MRE_medio", "COD_medio", "rank_medio_sMAPE"])
        for i, m in enumerate(methods):
            sm = np.mean([per_base[d][m]["sMAPE"] for d in bases])
            mr = np.mean([per_base[d][m]["MRE"] for d in bases])
            co = np.mean([per_base[d][m]["COD"] for d in bases])
            w.writerow([m, f"{sm:.3f}", f"{mr:.3f}", f"{co:.3f}", f"{avg[i]:.2f}"])

    # ---- teste estatístico (numérico) ----
    with open(os.path.join(RESULTS, "teste_friedman.txt"), "w", encoding="utf-8") as f:
        f.write(f"Teste de Friedman (metrica: sMAPE; {K} bases x {N} metodos)\n")
        f.write(f"  Estatistica F = {F:.4f}\n  p-valor       = {p:.6f}\n")
        f.write(f"  Significativo (p<0.05): {'SIM' if p < 0.05 else 'NAO'}\n\n")
        f.write(f"Post-hoc Bonferroni-Dunn (controle = {CONTROL})\n  CD = {cd:.4f}\n\n")
        f.write("Ranks medios (menor = melhor):\n")
        for i in np.argsort(avg):
            f.write(f"  {methods[i]:<14} {avg[i]:.3f}\n")
        cr = avg[methods.index(CONTROL)]
        f.write(f"\nDiferencas vs {CONTROL} (|diff| > CD = significativo):\n")
        for i in np.argsort(avg):
            if methods[i] == CONTROL: continue
            diff = abs(avg[i] - cr)
            f.write(f"  {methods[i]:<14} |diff|={diff:.3f}  {'SIGNIFICATIVO' if diff > cd else 'n.s.'}\n")

    _fig_cd(methods, avg, cd, p, K)
    _fig_ranks(methods, avg)
    _fig_pareto(bases, t4)
    _fig_smape(methods, bases, smape_mat)
    _fig_friedman_table(methods, bases, per_base, avg, F, p, cd)
    print(f"[make_results] OK -> {RESULTS}/ (tabelas, teste_friedman.txt) e {FIGS}/ (4 figuras)")
    print(f"  Friedman p={p:.3f} | CD={cd:.2f} | melhor rank: {methods[int(np.argmin(avg))]} ({avg.min():.2f})")


# ─── figuras ──────────────────────────────────────────────────────────────
def _fig_cd(methods, avg, cd, p, K):
    N = len(methods); order = np.argsort(avg); lo, hi = 1, N
    fig, ax = plt.subplots(figsize=(10, 2.8)); ax.set_xlim(lo - .3, hi + .3); ax.set_ylim(0, 1); ax.axis("off")
    ax.hlines(.75, lo, hi, color="#333")
    for x in range(1, N + 1):
        ax.vlines(x, .72, .78, color="#333"); ax.text(x, .83, str(x), ha="center", fontsize=8)
    cr = avg[methods.index(CONTROL)]
    ax.hlines(.62, cr, min(cr + cd, hi), color="#c0392b", lw=3)
    ax.text((cr + min(cr + cd, hi)) / 2, .55, f"CD={cd:.2f}", ha="center", color="#c0392b", fontsize=9)
    med = np.median(avg)
    left = [i for i in order if avg[i] <= med]; right = [i for i in order if avg[i] > med]
    for k, i in enumerate(left):
        y = .45 - k * .11; ax.plot([avg[i], avg[i]], [.72, y], color="#888", lw=.8); ax.plot([avg[i], lo - .2], [y, y], color="#888", lw=.8)
        c = "#c0392b" if methods[i] == CONTROL else "#2c3e50"
        ax.text(lo - .25, y, f"{methods[i]} ({avg[i]:.2f})", ha="right", va="center", fontsize=8, color=c)
    for k, i in enumerate(right[::-1]):
        y = .45 - k * .11; ax.plot([avg[i], avg[i]], [.72, y], color="#888", lw=.8); ax.plot([avg[i], hi + .2], [y, y], color="#888", lw=.8)
        c = "#c0392b" if methods[i] == CONTROL else "#2c3e50"
        ax.text(hi + .25, y, f"{methods[i]} ({avg[i]:.2f})", ha="left", va="center", fontsize=8, color=c)
    ax.set_title(f"Diagrama de Diferença Crítica (Friedman p={p:.3f}, {K} bases)\n"
                 f"Métodos a menos de CD do controle ({CONTROL}) não diferem significativamente", fontsize=9)
    plt.tight_layout(); plt.savefig(os.path.join(FIGS, "1_diagrama_diferenca_critica.png"), dpi=150, bbox_inches="tight"); plt.close()


def _fig_ranks(methods, avg):
    o = np.argsort(avg)[::-1]
    fig, ax = plt.subplots(figsize=(7, 5))
    cols = ["#c0392b" if methods[i] == CONTROL else "#5b8db8" for i in o]
    ax.barh([methods[i] for i in o], [avg[i] for i in o], color=cols)
    ax.set_xlabel("Rank médio por sMAPE (menor = melhor)")
    ax.set_title("Ranks médios (barra vermelha = método proposto)")
    plt.tight_layout(); plt.savefig(os.path.join(FIGS, "2_ranks_medios.png"), dpi=150, bbox_inches="tight"); plt.close()


def _fig_pareto(bases, t4):
    """Fronteira atingível: para cada nº de modelos, o melhor R² (OOF) possível.
    Com 5 modelos, enumeramos TODOS os subconjuntos (exato, não aproximado)."""
    fig, axs = plt.subplots(1, len(bases), figsize=(4 * len(bases), 3.4), sharey=True)
    for ax, d in zip(np.atleast_1d(axs), bases):
        oof = load_oof_train(d); _, _, y_train, _ = load_pool_arrays(d)
        M = oof.shape[1]
        ks, best = [], []
        for k in range(1, M + 1):
            r2k = max(pearson_r_squared(y_train, oof[:, list(c)].mean(1))
                      for c in itertools.combinations(range(M), k))
            ks.append(k); best.append(r2k)
        ax.plot(ks, best, "-o", color="#5b8db8", zorder=3)
        bb = t4[d]["best_balanced"]
        ax.scatter([bb["n_models"]], [bb["r2_train"]], color="#c0392b", s=110,
                   marker="*", zorder=5, label="escolha balanceada (GA)")
        ax.set_title(d); ax.set_xlabel("nº de modelos no ensemble")
        ax.set_xticks(range(1, M + 1)); ax.grid(alpha=.3)
    np.atleast_1d(axs)[0].set_ylabel("Melhor R² atingível (OOF)")
    np.atleast_1d(axs)[0].legend(fontsize=8, loc="lower right")
    fig.suptitle("Precisão × parcimônia: ganho de R² por modelo adicionado (NSGA-II escolhe o joelho da curva)", fontsize=11)
    plt.tight_layout(); plt.savefig(os.path.join(FIGS, "3_frente_pareto.png"), dpi=150, bbox_inches="tight"); plt.close()


def _fig_friedman_table(methods, bases, per_base, avg, F, p, cd):
    """Saída do Friedman/Bonferroni-Dunn como TABELA (imagem)."""
    order = np.argsort(avg)
    cr = avg[methods.index(CONTROL)]
    headers = ["Método", "Rank médio", "sMAPE méd.", "COD méd.", "|Δrank| vs controle", "Sig.?"]
    rows = []
    for i in order:
        sm = np.mean([per_base[d][methods[i]]["sMAPE"] for d in bases])
        co = np.mean([per_base[d][methods[i]]["COD"] for d in bases])
        if methods[i] == CONTROL:
            diff_s, sig = "— (controle)", "—"
        else:
            diff = abs(avg[i] - cr); diff_s = f"{diff:.2f}"; sig = "sim" if diff > cd else "não"
        rows.append([methods[i], f"{avg[i]:.2f}", f"{sm:.1f}", f"{co:.3f}", diff_s, sig])

    fig, ax = plt.subplots(figsize=(9, 0.5 + 0.42 * (len(rows) + 1)))
    ax.axis("off")
    ax.set_title(f"Teste de Friedman + post-hoc Bonferroni-Dunn  ({len(bases)} bases × {len(methods)} métodos)\n"
                 f"Estatística F = {F:.2f}   |   p-valor = {p:.3f}  ({'significativo' if p < 0.05 else 'não significativo'})"
                 f"   |   CD = {cd:.2f}", fontsize=10, pad=14)
    t = ax.table(cellText=rows, colLabels=headers, loc="center", cellLoc="center")
    t.auto_set_font_size(False); t.set_fontsize(9); t.scale(1, 1.5)
    for c in range(len(headers)):
        t[0, c].set_facecolor("#2c3e50"); t[0, c].set_text_props(color="white", fontweight="bold")
    for r, i in enumerate(order, start=1):
        if methods[i] == CONTROL:
            for c in range(len(headers)): t[r, c].set_facecolor("#fdecea")
    plt.tight_layout(); plt.savefig(os.path.join(FIGS, "5_tabela_friedman.png"), dpi=150, bbox_inches="tight"); plt.close()


def _fig_smape(methods, bases, smape_mat):
    key = ["Static", "SES-GA", "SES-GA-Multi", "DES"]; ki = [methods.index(k) for k in key]
    x = np.arange(len(bases)); w = .2
    fig, ax = plt.subplots(figsize=(9, 4))
    for j, (k, idx) in enumerate(zip(key, ki)):
        ax.bar(x + (j - 1.5) * w, [smape_mat[b, idx] for b in range(len(bases))], w, label=k)
    ax.set_xticks(x); ax.set_xticklabels(bases); ax.set_ylabel("sMAPE (%) — escala original")
    ax.legend(fontsize=8, ncol=4); ax.set_title("sMAPE por base — métodos de ensemble (menor = melhor)")
    plt.tight_layout(); plt.savefig(os.path.join(FIGS, "4_smape_por_base.png"), dpi=150, bbox_inches="tight"); plt.close()


if __name__ == "__main__":
    generate()
