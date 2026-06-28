"""
diagnostico_preproc.py — Avaliacao do pre-processamento das bases Finnish e
Maxwell. NAO altera o pipeline de replicacao; apenas analisa e gera um
relatorio (outputs/diagnostico_preprocessamento.md) + figuras de apoio.

Tecnicas usadas:
  - Distribuicao/assimetria do alvo (skew, kurtose) cru vs log1p  -> "desbalanceamento" em regressao
  - Ausentes / sentinelas
  - Variancia quase-nula (VarianceThreshold sobre features escaladas)
  - Redundancia: pares com |correlacao| alta
  - Multicolinearidade: VIF (1/(1-R^2) de cada feature contra as demais) e numero de condicao
  - Relevancia feature->alvo: informacao mutua
  - Dimensionalidade: razao amostras/atributos e PCA (componentes p/ 90% e 95% da variancia)
  - Outliers do alvo (regra IQR)
"""
import os
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.feature_selection import VarianceThreshold, mutual_info_regression
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

import data_prep as dp

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGDIR = os.path.join(ROOT, "outputs", "figures", "diagnostico")
os.makedirs(FIGDIR, exist_ok=True)


def carregar_selecionado(name):
    """Carrega a base com o MESMO recorte de features do pipeline, sem escalar."""
    cfg = dp.CONFIG[name]
    df = dp._load_raw(cfg)
    target = cfg["target"]
    if name == "finnish":
        df = df[df[target] > 0].reset_index(drop=True)
    df = df.dropna(subset=[target]).reset_index(drop=True)
    keep = [c for c in df.columns if c not in cfg["drop"]]
    sel = df[keep].copy()
    feat_cols = [c for c in keep if c != target]
    X = sel[feat_cols].astype(float)
    y = sel[target].astype(float).values
    return X, y, feat_cols, df, target


def vif_scores(Xs):
    """VIF de cada coluna: 1/(1-R^2) regredindo a coluna contra as demais."""
    vifs = {}
    cols = list(Xs.columns)
    for c in cols:
        others = [o for o in cols if o != c]
        if not others:
            vifs[c] = np.nan; continue
        r2 = r2_score(Xs[c], LinearRegression().fit(Xs[others], Xs[c]).predict(Xs[others]))
        vifs[c] = np.inf if r2 >= 1 else 1.0 / (1.0 - r2)
    return pd.Series(vifs)


def analisar(name):
    X, y, feat_cols, df_full, target = carregar_selecionado(name)
    n, p = X.shape
    out = {"name": name, "n": n, "p": p, "feat_cols": feat_cols}

    # --- alvo: distribuicao / "desbalanceamento" ---
    out["y_skew"] = float(skew(y))
    out["y_kurt"] = float(kurtosis(y))            # excess kurtosis
    ylog = np.log1p(y)
    out["ylog_skew"] = float(skew(ylog))
    out["ylog_kurt"] = float(kurtosis(ylog))
    q1, q3 = np.percentile(y, [25, 75]); iqr = q3 - q1
    out["y_outliers"] = int(np.sum((y < q1 - 1.5 * iqr) | (y > q3 + 1.5 * iqr)))
    out["y_max_over_median"] = float(np.max(y) / np.median(y))

    # --- ausentes / sentinelas ---
    out["n_missing_cells"] = int(X.isna().sum().sum())
    out["cols_com_zero"] = int((X == 0).any(axis=0).sum())

    # --- variancia quase-nula (sobre features escaladas MinMax) ---
    Xmm = pd.DataFrame(MinMaxScaler().fit_transform(X), columns=feat_cols)
    var = Xmm.var()
    out["near_zero_var"] = sorted(var[var < 0.01].index.tolist())
    out["const_feats"] = sorted(var[var == 0].index.tolist())

    # --- redundancia: pares de correlacao alta ---
    corr = X.corr().abs()
    iu = np.triu_indices(p, k=1)
    pares = [(feat_cols[i], feat_cols[j], corr.iloc[i, j])
             for i, j in zip(*iu)]
    out["pares_r_0_9"] = sorted([(a, b, round(r, 3)) for a, b, r in pares if r > 0.9],
                                key=lambda t: -t[2])
    out["n_pares_r_0_95"] = sum(1 for *_, r in [(a, b, r) for a, b, r in pares] if r > 0.95)

    # --- multicolinearidade: VIF e numero de condicao ---
    Xstd = pd.DataFrame(StandardScaler().fit_transform(X), columns=feat_cols)
    vifs = vif_scores(Xstd)
    out["vif_gt_10"] = sorted([(c, (round(v, 1) if np.isfinite(v) else "inf"))
                               for c, v in vifs.items() if v > 10],
                              key=lambda t: (t[1] == "inf", -(t[1] if t[1] != "inf" else 0)))
    out["n_vif_gt_10"] = int(np.sum(vifs > 10))
    svals = np.linalg.svd(Xstd.values, compute_uv=False)
    out["cond_number"] = float(svals.max() / svals[svals > 1e-12].min())

    # --- relevancia feature->alvo (informacao mutua) ---
    mi = mutual_info_regression(Xmm.values, y, random_state=42)
    mi = pd.Series(mi, index=feat_cols).sort_values(ascending=False)
    out["mi_top5"] = [(c, round(v, 3)) for c, v in mi.head(5).items()]
    out["mi_zero"] = sorted(mi[mi <= 1e-6].index.tolist())

    # --- dimensionalidade: razao e PCA ---
    out["ratio_n_p"] = round(n / p, 2)
    pca = PCA().fit(StandardScaler().fit_transform(X))
    cum = np.cumsum(pca.explained_variance_ratio_)
    out["pca_90"] = int(np.searchsorted(cum, 0.90) + 1)
    out["pca_95"] = int(np.searchsorted(cum, 0.95) + 1)
    out["pca_cum"] = cum

    _figuras(name, y, ylog, corr, cum)
    return out


def _figuras(name, y, ylog, corr, cum):
    # 1) distribuicao do alvo: cru vs log1p
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].hist(y, bins=30, color="#1f77b4"); ax[0].set_title(f"{name}: Effort (cru)")
    ax[0].set_xlabel("Effort"); ax[0].set_ylabel("freq")
    ax[1].hist(ylog, bins=30, color="#2ca02c"); ax[1].set_title(f"{name}: log1p(Effort)")
    ax[1].set_xlabel("log1p(Effort)")
    fig.tight_layout(); fig.savefig(os.path.join(FIGDIR, f"{name}_alvo_dist.png"), dpi=120)
    plt.close(fig)

    # 2) heatmap de correlacao das features
    fig, ax = plt.subplots(figsize=(9, 7.5))
    im = ax.imshow(corr.values, cmap="coolwarm", vmin=0, vmax=1)
    ax.set_xticks(range(len(corr))); ax.set_xticklabels(corr.columns, rotation=90, fontsize=6)
    ax.set_yticks(range(len(corr))); ax.set_yticklabels(corr.columns, fontsize=6)
    ax.set_title(f"{name}: |correlacao| entre features"); fig.colorbar(im, fraction=0.046)
    fig.tight_layout(); fig.savefig(os.path.join(FIGDIR, f"{name}_corr_heatmap.png"), dpi=120)
    plt.close(fig)

    # 3) scree PCA (variancia acumulada)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(range(1, len(cum) + 1), cum, marker="o", ms=3)
    ax.axhline(0.90, ls="--", c="gray"); ax.axhline(0.95, ls=":", c="gray")
    ax.set_xlabel("nº de componentes"); ax.set_ylabel("variancia acumulada")
    ax.set_title(f"{name}: PCA — variancia explicada acumulada"); ax.set_ylim(0, 1.02)
    fig.tight_layout(); fig.savefig(os.path.join(FIGDIR, f"{name}_pca_scree.png"), dpi=120)
    plt.close(fig)


def render_md(res):
    L = ["# Diagnóstico de Pré-processamento — Finnish e Maxwell",
         "",
         "Avaliação das duas bases **com o mesmo recorte de features do pipeline** "
         "(antes da normalização). Não altera a replicação; serve para decidir o "
         "que faz sentido melhorar. Figuras em `outputs/figures/diagnostico/`.",
         ""]
    for o in res:
        L += [f"## {o['name'].capitalize()}  ({o['n']} amostras × {o['p']} features, alvo incl. à parte)",
              "",
              "**Distribuição do alvo (desbalanceamento em regressão)**",
              "",
              f"- Skewness cru = **{o['y_skew']:.2f}**, kurtose (excess) = **{o['y_kurt']:.2f}** "
              f"→ fortemente assimétrico à direita. Após `log1p`: skew = **{o['ylog_skew']:.2f}**, "
              f"kurtose = **{o['ylog_kurt']:.2f}** (bem mais próximo do normal).",
              f"- Outliers do alvo (IQR): **{o['y_outliers']}** de {o['n']}. "
              f"Máx/mediana = **{o['y_max_over_median']:.1f}×**.",
              "",
              "**Ausentes / sentinelas**",
              "",
              f"- Células ausentes (NaN) no X selecionado: **{o['n_missing_cells']}**. "
              f"Colunas que contêm algum zero: {o['cols_com_zero']}.",
              "",
              "**Variância quase-nula**",
              "",
              f"- Features constantes: {o['const_feats'] or 'nenhuma'}.",
              f"- Variância quase-nula (<0.01 após MinMax): {o['near_zero_var'] or 'nenhuma'}.",
              "",
              "**Redundância (correlação alta entre features)**",
              "",
              f"- Pares com |r| > 0.95: **{o['n_pares_r_0_95']}**.",
              f"- Pares com |r| > 0.90 ({len(o['pares_r_0_9'])}): "
              f"{o['pares_r_0_9'][:8]}{' ...' if len(o['pares_r_0_9'])>8 else ''}",
              "",
              "**Multicolinearidade**",
              "",
              f"- Features com VIF > 10: **{o['n_vif_gt_10']}** "
              f"→ {o['vif_gt_10'][:8]}{' ...' if len(o['vif_gt_10'])>8 else ''}",
              f"- Número de condição da matriz (padronizada): **{o['cond_number']:.0f}** "
              f"({'alto → multicolinearidade' if o['cond_number']>30 else 'ok'}).",
              "",
              "**Relevância feature→alvo (informação mútua)**",
              "",
              f"- Top-5 informativas: {o['mi_top5']}.",
              f"- Sem informação (MI≈0): {o['mi_zero'] or 'nenhuma'}.",
              "",
              "**Dimensionalidade**",
              "",
              f"- Razão amostras/features = **{o['ratio_n_p']}**.",
              f"- PCA: **{o['pca_90']}** componentes para 90% da variância, "
              f"**{o['pca_95']}** para 95% (de {o['p']} features).",
              ""]
    path = os.path.join(ROOT, "outputs", "diagnostico_preprocessamento.md")
    with open(path, "w") as f:
        f.write("\n".join(L))
    return path


if __name__ == "__main__":
    res = [analisar("finnish"), analisar("maxwell")]
    path = render_md(res)
    # resumo no stdout
    for o in res:
        print(f"\n=== {o['name'].upper()} ({o['n']}x{o['p']}) ===")
        print(f"  alvo skew {o['y_skew']:.2f} -> log {o['ylog_skew']:.2f} | "
              f"outliers {o['y_outliers']} | max/median {o['y_max_over_median']:.1f}x")
        print(f"  near-zero-var: {o['near_zero_var']}")
        print(f"  pares |r|>0.9: {len(o['pares_r_0_9'])} | VIF>10: {o['n_vif_gt_10']} | "
              f"cond {o['cond_number']:.0f}")
        print(f"  MI zero: {o['mi_zero']}")
        print(f"  ratio n/p {o['ratio_n_p']} | PCA 95% -> {o['pca_95']}/{o['p']} comps")
    print("\nRelatorio:", path)
