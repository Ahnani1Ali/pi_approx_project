"""
main.py
───────
Script principal : exécute l'ensemble des expériences d'approximation de π
et génère tous les résultats (figures + CSV + rapport console).

Usage :
    python main.py              # toutes les expériences
    python main.py --fast       # mode rapide (hyperparamètres réduits)
    python main.py --method mc  # une seule méthode

Architecture :
    1. Génération du dataset
    2. Monte-Carlo (+ analyse de convergence + TCL)
    3. Régression polynomiale (+ sélection de degré + conditionnement)
    4. MLP from scratch (+ entraînement Adam + intégration)
    5. Processus Gaussien (+ MLE-II + incertitude bayésienne)
    6. Comparaison globale (+ Rademacher + bornes PAC)
    7. Sauvegarde des résultats (CSV + figures)
"""

import sys
import os
import time
import argparse
import numpy as np
import csv

# ── Ajout du répertoire racine au path ─────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from configs.config import (
    GLOBAL_SEED, DATASET, MONTE_CARLO, POLYNOMIAL,
    MLP as MLP_CFG, GAUSSIAN_PROCESS, COMPARISON, RADEMACHER,
    FIGURES_DIR, DATA_DIR
)
from src.dataset import generate_dataset, PI_REF
from src.monte_carlo import MonteCarloEstimator
from src.polynomial_regression import PolynomialPiEstimator
from src.neural_network import MLP
from src.gaussian_process import GaussianProcessRegressor
from src.rademacher import (
    compute_rademacher_vs_degree,
    compute_pac_vs_n,
    rademacher_polynomial,
    pac_bound,
)
from src.visualization import (
    plot_dataset, plot_monte_carlo, plot_polynomial,
    plot_mlp, plot_gp, plot_comparison, plot_rademacher,
    COLORS
)

os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
#  UTILITAIRES
# ══════════════════════════════════════════════════════════════════════════════

def banner(title: str) -> None:
    w = 65
    print("\n" + "═" * w)
    print(f"  {title}")
    print("═" * w)


def print_result(name: str, pi_hat: float, elapsed: float) -> None:
    err = abs(pi_hat - PI_REF)
    n_correct = max(0, -int(np.floor(np.log10(err)))) if err > 0 else 15
    print(f"  π̂  = {pi_hat:.12f}")
    print(f"  err = {err:.3e}   (~{n_correct} décimales correctes)")
    print(f"  t   = {elapsed:.2f}s")


def save_csv(rows: list[dict], filename: str) -> None:
    path = os.path.join(DATA_DIR, filename)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"  ✓  CSV sauvegardé : {path}")


# ══════════════════════════════════════════════════════════════════════════════
#  EXPÉRIENCES
# ══════════════════════════════════════════════════════════════════════════════

def run_dataset(fast: bool = False) -> object:
    """Génère et visualise le dataset."""
    banner("1. GÉNÉRATION DU DATASET")
    ds = generate_dataset(
        n=DATASET["n_train"],
        sigma_noise=DATASET["sigma_noise"],
        n_test=DATASET["n_test"],
        seed=GLOBAL_SEED,
    )
    print(ds.summary())

    plot_dataset(ds, save_path=os.path.join(FIGURES_DIR, "fig_dataset.png"))
    return ds


def run_monte_carlo(fast: bool = False) -> dict:
    """Expériences Monte-Carlo."""
    banner("2. MÉTHODE MONTE-CARLO")

    mc  = MonteCarloEstimator(seed=GLOBAL_SEED)
    ns  = MONTE_CARLO["n_samples_list"]

    # Convergence
    t0       = time.time()
    conv_res = mc.convergence_analysis(ns, seed=0)
    elapsed  = time.time() - t0

    best_n  = ns[-1]
    best_pi = conv_res["pi_hats"][-1]
    print_result("Monte-Carlo", best_pi, elapsed)

    # Validation TCL
    tcl_data = mc.validate_tcl(
        n_fixed=MONTE_CARLO["n_fixed_tcl"],
        n_simulations=MONTE_CARLO["n_bootstrap"],
    )
    ks = tcl_data["kolmogorov_smirnov"]
    print(f"\n  Validation TCL :")
    print(f"    σ_th = {tcl_data['sigma_theoretical']:.5f}  |  σ_emp = {tcl_data['std_sample']:.5f}")
    print(f"    KS p-value = {ks['p_value']:.4f}  (>0.05 → normalité non rejetée)")

    # Démo visuelle
    demo_res = mc.estimate(5000, seed=1, store_points=True)

    # Variable de contrôle
    pi_cv, var_cv = mc.control_variate_estimate(100_000)
    print(f"\n  Variable de contrôle : π̂_cv = {pi_cv:.8f}  |  err = {abs(pi_cv-PI_REF):.2e}")

    # Figures
    plot_monte_carlo(
        mc_conv=conv_res,
        tcl_data=tcl_data,
        demo_result=demo_res,
        save_path=os.path.join(FIGURES_DIR, "fig_montecarlo.png"),
    )

    # CSV
    rows = [
        {"n": int(n), "pi_hat": float(p), "error": float(e), "std": float(s)}
        for n, p, e, s in zip(
            conv_res["ns"], conv_res["pi_hats"],
            conv_res["errors"], conv_res["std_errors"]
        )
    ]
    save_csv(rows, "monte_carlo_convergence.csv")

    return {
        "pi_hat": best_pi,
        "error":  abs(best_pi - PI_REF),
        "time":   elapsed,
        "color":  COLORS["mc"],
        "conv":   conv_res,
    }


def run_polynomial(ds, fast: bool = False) -> dict:
    """Expériences régression polynomiale."""
    banner("3. RÉGRESSION POLYNOMIALE")

    t0  = time.time()
    est = PolynomialPiEstimator(
        max_degree=POLYNOMIAL["max_degree"],
        alpha_ridge=POLYNOMIAL["alpha_ridge"],
        cv_folds=POLYNOMIAL["cv_folds"],
        seed=GLOBAL_SEED,
    )
    res = est.fit(ds.X_train, ds.y_train)
    elapsed = time.time() - t0

    print(res)
    print_result("Poly.", res.pi_hat, elapsed)

    # Intégrale Gauss-Legendre (vérification)
    pi_gl = est.gauss_legendre_integral(n_points=100)
    print(f"\n  Intégrale Gauss-Legendre (vérif.) : π̂ = {pi_gl:.10f}")

    # Conditionnement
    cond = est.vandermonde_condition_number(ds.X_train)
    print(f"\n  Conditionnement κ(ΦᵀΦ) (d=2→8) :")
    for d, k in list(cond.items())[:7]:
        print(f"    d={d:2d}  κ = {k:.2e}")

    # Figures
    plot_polynomial(
        ds=ds, poly_est=est, poly_res=res, cond_numbers=cond,
        save_path=os.path.join(FIGURES_DIR, "fig_polynomial.png"),
    )

    # CSV scores CV
    rows = [{"degree": d, "mse_cv": v} for d, v in res.cv_scores.items()]
    save_csv(rows, "polynomial_cv_scores.csv")

    return {
        "pi_hat":     res.pi_hat,
        "error":      res.error_abs,
        "time":       elapsed,
        "best_degree": res.best_degree,
        "color":      COLORS["poly"],
    }


def run_mlp(ds, fast: bool = False) -> dict:
    """Expériences réseau de neurones MLP."""
    banner("4. RÉSEAU DE NEURONES MLP")

    n_epochs = 1000 if fast else MLP_CFG["n_epochs"]

    t0  = time.time()
    mlp = MLP(
        layer_sizes=MLP_CFG["layer_sizes"],
        activation="gelu",
        seed=GLOBAL_SEED,
    )
    res = mlp.fit_and_estimate(
        ds.X_train, ds.y_train,
        n_epochs=n_epochs,
        batch_size=MLP_CFG["batch_size"],
        lr=MLP_CFG["lr"],
        beta1=MLP_CFG["beta1"],
        beta2=MLP_CFG["beta2"],
        eps=MLP_CFG["eps_adam"],
        verbose_every=MLP_CFG["verbose_every"],
    )
    elapsed = time.time() - t0

    print(res)
    print_result("MLP", res.pi_hat, elapsed)
    print(f"\n  Paramètres : {mlp.n_parameters:,}")

    # Comparaison méthodes d'intégration
    pi_trapz   = mlp.estimate_pi(M=50_000, method="trapz")
    pi_simpson = mlp.estimate_pi(M=50_000, method="simpson")
    print(f"  Trapèzes  : π̂ = {pi_trapz:.10f}")
    print(f"  Simpson   : π̂ = {pi_simpson:.10f}")

    # Figures
    plot_mlp(
        ds=ds, mlp=mlp, mlp_res=res,
        save_path=os.path.join(FIGURES_DIR, "fig_mlp.png"),
    )

    # CSV perte
    rows = [{"epoch": e, "loss": l}
            for e, l in zip(res.history.epochs, res.history.train_losses)]
    save_csv(rows, "mlp_training_loss.csv")

    return {
        "pi_hat": res.pi_hat,
        "error":  res.error_abs,
        "time":   elapsed,
        "color":  COLORS["mlp"],
    }


def run_gp(ds, fast: bool = False) -> dict:
    """Expériences Processus Gaussien."""
    banner("5. PROCESSUS GAUSSIEN")

    n_gp = min(GAUSSIAN_PROCESS["n_gp_max"], ds.n)
    X_gp = ds.X_train[:n_gp]
    y_gp = ds.y_train[:n_gp]

    n_restarts = 3 if fast else GAUSSIAN_PROCESS["n_restarts"]

    t0 = time.time()
    gp = GaussianProcessRegressor(
        sigma_f=GAUSSIAN_PROCESS["sigma_f"],
        length_scale=GAUSSIAN_PROCESS["length_scale"],
        sigma_n=GAUSSIAN_PROCESS["sigma_n"],
    )
    res = gp.fit_and_estimate(
        X_gp, y_gp,
        optimize=GAUSSIAN_PROCESS["optimize"],
        n_restarts=n_restarts,
    )
    elapsed = time.time() - t0

    print(res)
    print_result("GP", res.pi_hat, elapsed)

    # Prédictions aux quintiles
    x_pts = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    mu, std = gp.predict(x_pts)
    print(f"\n  Prédictions aux points de test :")
    f_exact = np.sqrt(1 - x_pts**2)
    for x, m, s, fe in zip(x_pts, mu, std, f_exact):
        print(f"    f({x:.2f})={fe:.5f}  |  μ*={m:.5f} ± {2*s:.5f} (IC95%)")

    # Figures
    plot_gp(
        ds=ds, gp=gp, gp_res=res, n_gp=n_gp,
        save_path=os.path.join(FIGURES_DIR, "fig_gp.png"),
    )

    # CSV prédictions
    X_grid     = ds.X_test
    mu_g, s_g = gp.predict(X_grid)
    rows = [
        {"x": float(x), "mu": float(m), "sigma": float(s), "f_exact": float(fe)}
        for x, m, s, fe in zip(X_grid, mu_g, s_g, ds.y_test)
    ]
    save_csv(rows, "gp_posterior.csv")

    return {
        "pi_hat":  res.pi_hat,
        "pi_std":  res.pi_std,
        "error":   res.error_abs,
        "time":    elapsed,
        "color":   COLORS["gp"],
        "n_gp":    n_gp,
        "_gp_obj": gp,
    }


def run_comparison(ds, mc_res: dict, poly_res: dict,
                   mlp_res: dict, gp_res: dict, fast: bool = False) -> None:
    """Analyse comparative et figures globales."""
    banner("6. COMPARAISON GLOBALE")

    # Convergence multi-méthodes
    ns_comp = COMPARISON["ns_list"] if not fast else [50, 200, 1000]
    conv_data: dict = {"mc": {}, "poly": {}, "gp": {}}

    print("  Calcul des courbes de convergence...")
    mc  = MonteCarloEstimator(seed=GLOBAL_SEED)
    pi_mc_arr, pi_poly_arr, pi_gp_arr = [], [], []

    for n in ns_comp:
        # MC
        r = mc.estimate(n * 5, seed=0)
        pi_mc_arr.append(abs(r.pi_hat - PI_REF))

        # Poly
        try:
            ds_n = generate_dataset(n=n, sigma_noise=DATASET["sigma_noise"], seed=GLOBAL_SEED)
            ep   = PolynomialPiEstimator(max_degree=10, alpha_ridge=1e-12, cv_folds=3)
            er   = ep.fit(ds_n.X_train, ds_n.y_train)
            pi_poly_arr.append(abs(er.pi_hat - PI_REF))
        except Exception:
            pi_poly_arr.append(np.nan)

        # GP
        try:
            n_gp = min(n, 150)
            ds_g = generate_dataset(n=n_gp, sigma_noise=DATASET["sigma_noise"], seed=GLOBAL_SEED)
            gp_c = GaussianProcessRegressor()
            gp_c.fit(ds_g.X_train, ds_g.y_train, optimize=False)
            pi_g, _ = gp_c.estimate_pi(M=5000)
            pi_gp_arr.append(abs(pi_g - PI_REF))
        except Exception:
            pi_gp_arr.append(np.nan)

    conv_data["mc"]   = {"ns": np.array(ns_comp) * 5, "errors": np.array(pi_mc_arr)}
    conv_data["poly"] = {"ns": np.array(ns_comp),     "errors": np.array(pi_poly_arr)}
    conv_data["gp"]   = {"ns": np.array(ns_comp),     "errors": np.array(pi_gp_arr)}

    # Résultats finaux
    results_dict = {
        "MC":   {"pi": mc_res["pi_hat"],   "error": mc_res["error"],   "time": mc_res["time"],   "color": COLORS["mc"]},
        "Poly": {"pi": poly_res["pi_hat"], "error": poly_res["error"], "time": poly_res["time"], "color": COLORS["poly"]},
        "MLP":  {"pi": mlp_res["pi_hat"],  "error": mlp_res["error"],  "time": mlp_res["time"],  "color": COLORS["mlp"]},
        "GP":   {"pi": gp_res["pi_hat"],   "error": gp_res["error"],   "time": gp_res["time"],   "color": COLORS["gp"]},
    }

    plot_comparison(
        results_dict=results_dict,
        conv_data=conv_data,
        save_path=os.path.join(FIGURES_DIR, "fig_comparison.png"),
    )

    # Rademacher
    banner("7. COMPLEXITÉ DE RADEMACHER")
    ns_rad  = RADEMACHER["ns_list"]
    degrees = RADEMACHER["degrees"]
    rad_data = {"ns_list": ns_rad, "degrees": degrees}
    for n_r in ns_rad:
        ds_r = generate_dataset(n=n_r, sigma_noise=0, seed=GLOBAL_SEED)
        vals = [rademacher_polynomial(ds_r.X_train, d, n_mc=200) for d in degrees]
        rad_data[f"n={n_r}"] = vals

    ns_pac = [50, 100, 200, 500] if not fast else [50, 100]
    pac_data = []
    for n_p in ns_pac:
        ds_p = generate_dataset(n=n_p, sigma_noise=DATASET["sigma_noise"], seed=GLOBAL_SEED)
        r = pac_bound(ds_p.X_train, ds_p.y_train, degree=7, delta=0.05, n_mc_rad=150)
        pac_data.append(r)
        print(f"  n={n_p:4d}  bound={r['bound']:.3e}  emp={r['emp']:.3e}  "
              f"2𝔑̂={2*r['rad']:.3e}  conf={r['conf']:.3e}")

    plot_rademacher(
        rad_data=rad_data,
        pac_data=pac_data,
        save_path=os.path.join(FIGURES_DIR, "fig_rademacher.png"),
    )

    # Sauvegarde CSV des convergences
    rows_conv = []
    for i, n in enumerate(ns_comp):
        rows_conv.append({
            "n_poly_gp": n,
            "n_mc": n * 5,
            "error_mc":   float(pi_mc_arr[i]),
            "error_poly": float(pi_poly_arr[i]),
            "error_gp":   float(pi_gp_arr[i]),
        })
    save_csv(rows_conv, "convergence_comparison.csv")

    # CSV résultats finaux
    rows_final = [
        {"method": k, "pi_hat": v["pi"], "error": v["error"], "time": v["time"]}
        for k, v in results_dict.items()
    ]
    save_csv(rows_final, "final_results.csv")


def print_final_summary(mc_res, poly_res, mlp_res, gp_res) -> None:
    """Affiche le résumé final comparatif."""
    banner("RÉSUMÉ FINAL — APPROXIMATION DE π PAR ML")
    print(f"  Valeur exacte : π = {PI_REF:.15f}\n")
    results = [
        ("Monte-Carlo",          mc_res["pi_hat"],   mc_res["time"]),
        (f"Poly. (d={poly_res.get('best_degree','?')})",
                                 poly_res["pi_hat"], poly_res["time"]),
        ("MLP (GELU, Adam)",     mlp_res["pi_hat"],  mlp_res["time"]),
        ("Processus Gaussien",   gp_res["pi_hat"],   gp_res["time"]),
    ]
    leibniz = 4 * sum((-1)**k / (2*k+1) for k in range(100_000))
    results.append(("Leibniz (100k termes)", leibniz, 0.0))

    fmt = "  {:<28} π̂={:.10f}   err={:.2e}   t={:.2f}s"
    for name, pi_hat, t in results:
        err = abs(pi_hat - PI_REF)
        print(fmt.format(name, pi_hat, err, t))

    print("\n  ✓ Toutes les figures → results/figures/")
    print("  ✓ Tous les CSV      → results/data/")
    print("═" * 65)


# ══════════════════════════════════════════════════════════════════════════════
#  POINT D'ENTRÉE
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(
        description="Approximation de π par Machine Learning"
    )
    parser.add_argument("--fast",   action="store_true",
                        help="Mode rapide (hyperparamètres réduits)")
    parser.add_argument("--method", choices=["all", "mc", "poly", "mlp", "gp"],
                        default="all", help="Méthode à exécuter")
    return parser.parse_args()


def main():
    args = parse_args()
    fast = args.fast

    print("\n" + "█" * 65)
    print("  PROJET M1 — APPROXIMATION DE π PAR MACHINE LEARNING")
    print(f"  Mode : {'RAPIDE' if fast else 'COMPLET'}")
    print("█" * 65)

    ds       = run_dataset(fast)
    mc_res   = None
    poly_res = None
    mlp_res  = None
    gp_res   = None

    if args.method in ("all", "mc"):
        mc_res = run_monte_carlo(fast)

    if args.method in ("all", "poly"):
        poly_res = run_polynomial(ds, fast)

    if args.method in ("all", "mlp"):
        mlp_res = run_mlp(ds, fast)

    if args.method in ("all", "gp"):
        gp_res = run_gp(ds, fast)

    if args.method == "all":
        run_comparison(ds, mc_res, poly_res, mlp_res, gp_res, fast)
        print_final_summary(mc_res, poly_res, mlp_res, gp_res)


if __name__ == "__main__":
    main()
