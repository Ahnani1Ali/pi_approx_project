"""
src/visualization.py
─────────────────────
Module de visualisation pour tous les résultats d'approximation de π.
Génère des figures publication-quality (300 dpi).

Figures produites :
    fig_dataset.png      — Dataset et interprétation géométrique
    fig_montecarlo.png   — Points MC, convergence, validation TCL
    fig_polynomial.png   — Ajustement, sélection de degré, convergence
    fig_mlp.png          — Perte, ajustement, résidus MLP
    fig_gp.png           — Postérieure GP, incertitude, tirages
    fig_comparison.png   — Comparaison globale des méthodes
    fig_rademacher.png   — Complexité de Rademacher et bornes PAC
    fig_convergence.png  — Courbes de convergence comparatives
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import LogLocator
from scipy.stats import norm as scipy_norm
from typing import Optional
import os

# ── Palette de couleurs ────────────────────────────────────────────────────────
COLORS = {
    "mc":   "#1f4e79",
    "poly": "#1d6b2e",
    "mlp":  "#8b0000",
    "gp":   "#d4700a",
    "true": "#444444",
    "ref":  "#999999",
}

PI_REF = np.pi


def _setup_style():
    """Configure le style matplotlib global."""
    plt.rcParams.update({
        "figure.dpi":          130,
        "font.family":         "DejaVu Sans",
        "font.size":           10,
        "axes.titlesize":      11,
        "axes.titleweight":    "bold",
        "axes.spines.top":     False,
        "axes.spines.right":   False,
        "axes.grid":           True,
        "grid.alpha":          0.25,
        "grid.linestyle":      "--",
        "legend.framealpha":   0.85,
        "legend.fontsize":     8,
    })

_setup_style()


def save_fig(fig: plt.Figure, path: str, dpi: int = 180) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, bbox_inches="tight", dpi=dpi)
    plt.close(fig)
    print(f"  ✓  Sauvegardé : {path}")


# ══════════════════════════════════════════════════════════════════════════════
#  FIGURE 1 : Dataset
# ══════════════════════════════════════════════════════════════════════════════

def plot_dataset(ds, save_path: Optional[str] = None) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    ax = axes[0]
    ax.scatter(ds.X_train, ds.y_train, s=8, alpha=0.45,
               color=COLORS["mc"], label=f"Données bruitées ($n$={ds.n}, $\\sigma$={ds.sigma_noise})")
    ax.plot(ds.X_test, ds.y_test, color=COLORS["true"], lw=2.2,
            label=r"$f(x)=\sqrt{1-x^2}$ (exacte)")
    ax.set(xlabel="$x$", ylabel="$y$",
           title="Dataset synthétique : quart de cercle")
    ax.legend()
    ax.set_aspect("equal")

    ax = axes[1]
    theta = np.linspace(0, np.pi / 2, 600)
    ax.fill_between(np.cos(theta), np.sin(theta), alpha=0.18,
                    color=COLORS["gp"],
                    label=f"Aire = $\\pi/4 \\approx$ {PI_REF/4:.5f}")
    ax.plot(np.cos(theta), np.sin(theta), color=COLORS["mc"], lw=2.5,
            label="Quart de cercle")
    ax.plot([0, 1, 1, 0, 0], [0, 0, 1, 1, 0], "k--", lw=1, alpha=0.4)
    ax.text(0.2, 0.35, r"$\int_0^1 \sqrt{1-x^2}\,dx = \frac{\pi}{4}$",
            fontsize=13, color=COLORS["mc"],
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    ax.set_aspect("equal")
    ax.set_xlim(-0.05, 1.1); ax.set_ylim(-0.05, 1.1)
    ax.set(xlabel="$x$", ylabel="$y$",
           title=r"Interprétation géométrique de $\pi$")
    ax.legend()

    fig.suptitle("Jeu de données synthétique pour l'approximation de $\\pi$",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    if save_path:
        save_fig(fig, save_path)
    return fig


# ══════════════════════════════════════════════════════════════════════════════
#  FIGURE 2 : Monte-Carlo
# ══════════════════════════════════════════════════════════════════════════════

def plot_monte_carlo(mc_conv: dict, tcl_data: dict, demo_result,
                     save_path: Optional[str] = None) -> plt.Figure:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))

    # (a) Points MC
    ax = axes[0]
    pts    = demo_result.points
    inside = demo_result.inside
    ax.scatter(pts[inside, 0],  pts[inside, 1],  s=1.8, alpha=0.4,
               color=COLORS["mc"], label="Dans le cercle")
    ax.scatter(pts[~inside, 0], pts[~inside, 1], s=1.8, alpha=0.4,
               color="#e05c5c", label="Hors du cercle")
    theta = np.linspace(0, np.pi / 2, 400)
    ax.plot(np.cos(theta), np.sin(theta), "k-", lw=2)
    ax.set_aspect("equal")
    ax.set(xlabel="$x$", ylabel="$y$",
           title=f"Monte-Carlo ($n$={demo_result.n:,})\n$\\hat{{\\pi}}$={demo_result.pi_hat:.5f}")
    ax.legend(fontsize=8, markerscale=4)

    # (b) Convergence log-log
    ax = axes[1]
    ns   = mc_conv["ns"]
    errs = mc_conv["errors"]
    stds = mc_conv["std_errors"]
    ax.loglog(ns, errs, color=COLORS["mc"], lw=1.8, label="Erreur MC")
    ax.fill_between(ns, np.maximum(errs - stds, 1e-9), errs + stds,
                    alpha=0.2, color=COLORS["mc"])
    ref_x = np.array([ns[0], ns[-1]], dtype=float)
    ax.loglog(ref_x, 2.0 * ref_x**(-0.5), "k--", lw=1.5, label=r"$O(n^{-1/2})$")
    ax.set(xlabel="$n$", ylabel=r"$|\hat{\pi}_n - \pi|$",
           title="Convergence Monte-Carlo (log-log)")
    ax.legend()

    # (c) Validation TCL
    ax = axes[2]
    samples  = tcl_data["samples"]
    sigma_th = tcl_data["sigma_theoretical"]
    ax.hist(samples, bins=55, density=True, color=COLORS["mc"],
            alpha=0.65, edgecolor="white")
    x_g = np.linspace(samples.min(), samples.max(), 300)
    ax.plot(x_g, scipy_norm.pdf(x_g, PI_REF, sigma_th), "r-", lw=2,
            label=r"$\mathcal{N}(\pi,\,\sigma^2_{th})$ (TCL)")
    ax.axvline(PI_REF, color="k", lw=1.8, linestyle="--",
               label=f"$\\pi$ = {PI_REF:.5f}")
    ax.set(xlabel=r"$\hat{\pi}_n$", ylabel="Densité",
           title=f"Distribution de $\\hat{{\\pi}}_n$ (TCL, $n$={tcl_data['n_fixed']:,})")
    ax.legend()
    p_val = tcl_data["kolmogorov_smirnov"]["p_value"]
    ax.text(0.02, 0.96, f"KS p-value = {p_val:.3f}",
            transform=ax.transAxes, fontsize=8, va="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.6))

    fig.suptitle("Méthode Monte-Carlo — Convergence et validation du TCL",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    if save_path:
        save_fig(fig, save_path)
    return fig


# ══════════════════════════════════════════════════════════════════════════════
#  FIGURE 3 : Régression Polynomiale
# ══════════════════════════════════════════════════════════════════════════════

def plot_polynomial(ds, poly_est, poly_res, cond_numbers: dict,
                    save_path: Optional[str] = None) -> plt.Figure:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))

    X_test = ds.X_test
    y_pred = poly_est.predict(X_test)

    # (a) Ajustement
    ax = axes[0]
    ax.scatter(ds.X_train, ds.y_train, s=8, alpha=0.4, color=COLORS["poly"])
    ax.plot(X_test, ds.y_test, color=COLORS["true"], lw=2.2, label="Exacte")
    ax.plot(X_test, y_pred, "--", color=COLORS["poly"], lw=2,
            label=f"Poly. $d$={poly_res.best_degree}")
    ax.fill_between(X_test, ds.y_test, y_pred, alpha=0.25,
                    color=COLORS["poly"], label="Résidu")
    ax.set(xlabel="$x$", ylabel="$y$",
           title=f"Ajustement polynomial ($d$={poly_res.best_degree})\n$\\hat{{\\pi}}$={poly_res.pi_hat:.8f}")
    ax.legend()

    # (b) MSE CV vs degré
    ax = axes[1]
    degrees  = list(poly_res.cv_scores.keys())
    mse_vals = [poly_res.cv_scores[d] for d in degrees]
    ax.semilogy(degrees, mse_vals, "o-", color=COLORS["poly"], lw=2, ms=5)
    ax.axvline(poly_res.best_degree, color="red", lw=1.8, linestyle="--",
               label=f"Optimal $d$={poly_res.best_degree}")
    ax.set(xlabel="Degré $d$", ylabel="MSE CV (log)",
           title="Sélection du degré (validation croisée 5-fold)")
    ax.legend()

    # (c) Conditionnement de Vandermonde
    ax = axes[2]
    d_vals = list(cond_numbers.keys())
    c_vals = [cond_numbers[d] for d in d_vals]
    ax.semilogy(d_vals, c_vals, "s-", color=COLORS["mlp"], lw=2, ms=5,
                label=r"$\kappa(\Phi^\top\Phi)$")
    ax.axvline(poly_res.best_degree, color="red", lw=1.5, linestyle="--",
               alpha=0.7, label=f"$d$ optimal = {poly_res.best_degree}")
    ax.set(xlabel="Degré $d$",
           ylabel=r"$\kappa(\Phi^\top\Phi)$ (log)",
           title="Conditionnement de la matrice de Vandermonde")
    ax.legend()
    ax.text(0.5, 0.08, "Ridge nécessaire si κ ≫ 1",
            transform=ax.transAxes, ha="center", fontsize=8,
            bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))

    fig.suptitle("Régression Polynomiale Ridge — Ajustement et analyse",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    if save_path:
        save_fig(fig, save_path)
    return fig


# ══════════════════════════════════════════════════════════════════════════════
#  FIGURE 4 : MLP
# ══════════════════════════════════════════════════════════════════════════════

def plot_mlp(ds, mlp, mlp_res, save_path: Optional[str] = None) -> plt.Figure:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))

    X_test = ds.X_test
    y_pred = mlp.predict(X_test)
    resids = y_pred - ds.y_test

    # (a) Courbe de perte
    ax = axes[0]
    epochs = mlp_res.history.epochs
    losses = mlp_res.history.train_losses
    ax.semilogy(epochs, losses, color=COLORS["mlp"], lw=1.5, alpha=0.85)
    ax.set(xlabel="Époque", ylabel="MSE Loss (log)",
           title=f"Courbe de perte (Adam)\nLoss finale : {losses[-1]:.2e}")
    ax.text(0.6, 0.85, f"n_params = {mlp.n_parameters:,}",
            transform=ax.transAxes, fontsize=8,
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.6))

    # (b) Ajustement
    ax = axes[1]
    ax.scatter(ds.X_train, ds.y_train, s=8, alpha=0.4, color=COLORS["mlp"])
    ax.plot(X_test, ds.y_test, color=COLORS["true"], lw=2.2, label="Exacte")
    ax.plot(X_test, y_pred, "--", color=COLORS["mlp"], lw=2,
            label=f"MLP (GELU, {len(mlp.layer_sizes)-2} couches)")
    ax.fill_between(X_test, ds.y_test, y_pred, alpha=0.25,
                    color=COLORS["mlp"], label="Résidu")
    arch = " → ".join(str(s) for s in mlp.layer_sizes)
    ax.set(xlabel="$x$", ylabel="$y$",
           title=f"Ajustement MLP [{arch}]\n$\\hat{{\\pi}}$={mlp_res.pi_hat:.8f}")
    ax.legend()

    # (c) Distribution des résidus
    ax = axes[2]
    ax.hist(resids, bins=45, density=True, color=COLORS["mlp"],
            alpha=0.65, edgecolor="white")
    x_r = np.linspace(resids.min(), resids.max(), 200)
    ax.plot(x_r, scipy_norm.pdf(x_r, resids.mean(), resids.std()),
            "r-", lw=2, label=r"$\mathcal{N}(\mu,\sigma^2)$")
    ax.axvline(0, color="k", lw=1.5, linestyle="--")
    ax.set(xlabel="Résidu $\\hat{f}(x)-f(x)$", ylabel="Densité",
           title=f"Distribution des résidus\n$\\mu$={resids.mean():.2e}, $\\sigma$={resids.std():.2e}")
    ax.legend()

    fig.suptitle("Réseau de Neurones MLP — Entraînement et prédiction",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    if save_path:
        save_fig(fig, save_path)
    return fig


# ══════════════════════════════════════════════════════════════════════════════
#  FIGURE 5 : Processus Gaussien
# ══════════════════════════════════════════════════════════════════════════════

def plot_gp(ds, gp, gp_res, n_gp: int, save_path: Optional[str] = None) -> plt.Figure:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))

    X_test = ds.X_test
    mu, std = gp.predict(X_test)

    X_gp = ds.X_train[:n_gp]
    y_gp = ds.y_train[:n_gp]

    # (a) Postérieure avec bandes
    ax = axes[0]
    ax.scatter(X_gp, y_gp, s=10, zorder=5, color=COLORS["gp"], alpha=0.55)
    ax.plot(X_test, ds.y_test, color=COLORS["true"], lw=2.2, label="Exacte")
    ax.plot(X_test, mu, color=COLORS["gp"], lw=2.5, label="Moyenne $\\mu_*(x)$")
    ax.fill_between(X_test, mu - 2*std, mu + 2*std,
                    alpha=0.2, color=COLORS["gp"], label="IC 95% ($\\pm 2\\sigma_*$)")
    ax.fill_between(X_test, mu - std, mu + std,
                    alpha=0.35, color=COLORS["gp"])
    ax.set(xlabel="$x$", ylabel="$y$",
           title=f"GP Postérieure\n$\\hat{{\\pi}}$={gp_res.pi_hat:.8f}")
    ax.legend(loc="lower left")

    # (b) Incertitude σ*(x)
    ax = axes[1]
    ax.plot(X_test, std, color=COLORS["gp"], lw=2.2)
    ax.fill_between(X_test, 0, std, alpha=0.3, color=COLORS["gp"])
    ax.scatter(X_gp, np.full_like(X_gp, std.min() * 0.1),
               s=8, color=COLORS["gp"], alpha=0.4, zorder=5,
               label="Points d'entraînement")
    ax.set(xlabel="$x$",
           ylabel="$\\sigma_*(x)$ (écart-type postérieur)",
           title="Incertitude épistémique du GP")
    ax.legend()

    # (c) Tirages de la postérieure
    ax = axes[2]
    samples = gp.sample_posterior(X_test, n_samples=8, seed=42)
    for s in samples:
        ax.plot(X_test, s, alpha=0.35, lw=0.9, color=COLORS["gp"])
    ax.plot(X_test, mu, color="black", lw=2, label="Moyenne $\\mu_*(x)$")
    ax.plot(X_test, ds.y_test, "--", color=COLORS["true"], lw=1.5,
            label="$f(x)$ exacte")
    ax.set(xlabel="$x$", ylabel="$y$",
           title="Tirages de la distribution postérieure")
    ax.legend()
    ax.text(0.02, 0.04, f"σ_f={gp_res.sigma_f:.3f}  ℓ={gp_res.length_scale:.3f}",
            transform=ax.transAxes, fontsize=8,
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.7))

    fig.suptitle("Processus Gaussien — Inférence bayésienne de $\\pi$",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    if save_path:
        save_fig(fig, save_path)
    return fig


# ══════════════════════════════════════════════════════════════════════════════
#  FIGURE 6 : Comparaison globale
# ══════════════════════════════════════════════════════════════════════════════

def plot_comparison(results_dict: dict, conv_data: dict,
                    save_path: Optional[str] = None) -> plt.Figure:
    fig = plt.figure(figsize=(16, 9))
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    # ── (1) Convergence log-log (haut, pleine largeur) ────────────────────────
    ax1 = fig.add_subplot(gs[0, :])
    for key, label, color, ls in [
        ("mc",   "Monte-Carlo",          COLORS["mc"],   "o-"),
        ("poly", "Régression Poly.",      COLORS["poly"], "s-"),
        ("gp",   "Processus Gaussien",   COLORS["gp"],   "d-"),
    ]:
        if key in conv_data and len(conv_data[key]["errors"]) > 0:
            ns   = conv_data[key]["ns"]
            errs = conv_data[key]["errors"]
            ax1.loglog(ns, errs, ls, color=color, lw=2, ms=7, label=label)

    ref_x = np.logspace(1.5, 4, 50)
    ax1.loglog(ref_x, 3.0 * ref_x**(-0.5), "--", color=COLORS["mc"],
               alpha=0.5, lw=1.5, label=r"$O(n^{-1/2})$")
    ax1.loglog(ref_x, 2e-1 * ref_x**(-1.8), "--", color=COLORS["poly"],
               alpha=0.5, lw=1.5, label=r"$O(n^{-1.8})$")
    ax1.set(xlabel="$n$",
            ylabel=r"$|\hat{\pi}-\pi|$",
            title="Taux de convergence comparatifs (log-log)")
    ax1.legend(ncol=3)
    ax1.axhline(1e-4, color="gray", lw=0.8, linestyle=":", alpha=0.7)
    ax1.text(ref_x[-1]*0.7, 1.4e-4, "$10^{-4}$", fontsize=8, color="gray")

    # ── (2) Barres d'erreur finale ────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1, 0])
    names  = list(results_dict.keys())
    errors = [results_dict[k]["error"] for k in names]
    colors = [results_dict[k]["color"] for k in names]
    bars   = ax2.bar(range(len(names)), errors, color=colors, alpha=0.82,
                     edgecolor="white", width=0.6)
    ax2.set_yscale("log")
    ax2.set_xticks(range(len(names)))
    ax2.set_xticklabels(names, fontsize=8)
    ax2.set(ylabel=r"$|\hat{\pi}-\pi|$ (log)", title="Erreur absolue finale")
    for bar, e in zip(bars, errors):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height()*1.4,
                 f"{e:.1e}", ha="center", va="bottom", fontsize=7.5, rotation=15)

    # ── (3) Pareto précision / temps ──────────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 1])
    for name, rd in results_dict.items():
        ax3.scatter(rd["time"], rd["error"], s=120, color=rd["color"],
                    zorder=5, edgecolors="white", linewidths=1.5)
        ax3.annotate(name, (rd["time"], rd["error"]),
                     textcoords="offset points", xytext=(6, 4), fontsize=7.5)
    ax3.set_xscale("log"); ax3.set_yscale("log")
    ax3.set(xlabel="Temps CPU (s, log)",
            ylabel=r"$|\hat{\pi}-\pi|$ (log)",
            title="Frontière de Pareto\nPrécision vs Coût")

    # ── (4) Tableau récap ─────────────────────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 2])
    ax4.axis("off")
    table_data = [
        ["Méthode", "π̂", "Erreur", "Taux"],
        ["Monte-Carlo", f"{results_dict['MC']['pi']:.7f}",
         f"{results_dict['MC']['error']:.1e}", r"$O(n^{-1/2})$"],
        ["Poly.",   f"{results_dict['Poly']['pi']:.7f}",
         f"{results_dict['Poly']['error']:.1e}", r"$O(n^{-1.8})$"],
        ["MLP",    f"{results_dict['MLP']['pi']:.7f}",
         f"{results_dict['MLP']['error']:.1e}", "empirique"],
        ["GP",     f"{results_dict['GP']['pi']:.7f}",
         f"{results_dict['GP']['error']:.1e}", r"$O(n^{-2})$"],
    ]
    tbl = ax4.table(cellText=table_data[1:], colLabels=table_data[0],
                    cellLoc="center", loc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1.2, 1.8)
    for (r, c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_facecolor(COLORS["mc"])
            cell.set_text_props(color="white", fontweight="bold")
        elif r % 2 == 0:
            cell.set_facecolor("#f0f4f8")
    ax4.set_title("Tableau comparatif", fontweight="bold", pad=15)

    fig.suptitle(r"Comparaison globale : Approximation de $\pi$ par ML",
                 fontsize=14, fontweight="bold")
    if save_path:
        save_fig(fig, save_path)
    return fig


# ══════════════════════════════════════════════════════════════════════════════
#  FIGURE 7 : Rademacher
# ══════════════════════════════════════════════════════════════════════════════

def plot_rademacher(rad_data: dict, pac_data: dict,
                    save_path: Optional[str] = None) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # (a) Complexité de Rademacher vs degré
    ax = axes[0]
    for n, ls in zip(rad_data["ns_list"], ["-", "--", ":"]):
        key = f"n={n}"
        if key in rad_data:
            ax.plot(rad_data["degrees"], rad_data[key], ls, lw=2, label=f"$n$={n}")
    ax.set(xlabel="Degré $d$",
           ylabel=r"$\hat{\mathfrak{R}}_n(\mathcal{P}_d)$",
           title="Complexité de Rademacher\nvs degré polynomial")
    ax.legend()

    # (b) Décomposition de la borne PAC
    ax = axes[1]
    ns      = [d["n"] for d in pac_data]
    bounds  = [d["bound"]   for d in pac_data]
    emp_r   = [d["emp"]     for d in pac_data]
    rad_2   = [2*d["rad"]   for d in pac_data]
    conf    = [d["conf"]    for d in pac_data]

    ax.loglog(ns, bounds, "o-",  color=COLORS["poly"],  lw=2, ms=8, label="Borne PAC")
    ax.loglog(ns, emp_r,  "s--", color=COLORS["mc"],    lw=2, ms=6, label=r"$\hat{R}_n$ empirique")
    ax.loglog(ns, rad_2,  "^--", color=COLORS["gp"],    lw=2, ms=6, label=r"$2\hat{\mathfrak{R}}_n$")
    ax.loglog(ns, conf,   "v--", color=COLORS["mlp"],   lw=2, ms=6,
              label=r"$\sqrt{\log(1/\delta)/2n}$")
    ax.set(xlabel="$n$", ylabel="Valeur (log)",
           title="Décomposition borne PAC\n(Bartlett & Mendelson 2002)")
    ax.legend(fontsize=7.5)

    fig.suptitle("Théorie de l'apprentissage — Complexité de Rademacher et borne PAC",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    if save_path:
        save_fig(fig, save_path)
    return fig
