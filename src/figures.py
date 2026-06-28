"""
figures.py — Figuras 4, 5 (scatter SES/DES/OES) e 6, 7 (radar sMAPE vs COD).
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

COLORS = {"SES": "#1f77b4", "DES": "#d62728", "OES": "#2ca02c"}


def _trend(ax, x, y, color):
    """Reta de tendencia linear (least squares)."""
    if len(x) >= 2 and np.std(x) > 0:
        a, b = np.polyfit(x, y, 1)
        xs = np.linspace(min(x), max(x), 100)
        ax.plot(xs, a * xs + b, color=color, lw=1.5)


def scatter_fig(path, title, ens_train, ens_test, y_train, y_test):
    """Fig 4/5: (a) Training e (b) Testing — Actual (x) vs Predicted (y)."""
    fig, axes = plt.subplots(2, 1, figsize=(7, 10))
    for ax, (subtitle, ens, ytrue) in zip(
            axes, [("(a) Training", ens_train, y_train),
                   ("(b) Testing", ens_test, y_test)]):
        for name in ["SES", "DES", "OES"]:
            ax.scatter(ytrue, ens[name], s=22, alpha=0.7,
                       color=COLORS[name], label=name)
            _trend(ax, ytrue, ens[name], COLORS[name])
        ax.set_xlabel("Actual"); ax.set_ylabel("Predicted")
        ax.set_title(subtitle); ax.legend(loc="upper left", fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def radar_fig(path, title, model_order, smape_vals, cod_vals):
    """Fig 6/7: radar com um eixo por modelo; series sMAPE e COD.

    COD (em [0,1]) e reescalado para 0..100 para co-plotar com sMAPE (%).
    """
    labels = model_order
    n = len(labels)
    ang = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    ang += ang[:1]
    s = list(smape_vals) + [smape_vals[0]]
    c = list(np.array(cod_vals) * 100.0) + [cod_vals[0] * 100.0]

    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, polar=True)
    ax.plot(ang, s, color="#1f77b4", lw=1.8, label="sMAPE")
    ax.fill(ang, s, color="#1f77b4", alpha=0.08)
    ax.plot(ang, c, color="#ff7f0e", lw=1.8, label="COD (x100)")
    ax.fill(ang, c, color="#ff7f0e", alpha=0.08)
    ax.set_xticks(ang[:-1]); ax.set_xticklabels(labels, fontsize=8)
    ax.set_title(title); ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.1))
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
