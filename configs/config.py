"""
configs/config.py
─────────────────
Paramètres globaux du projet : seeds, tailles de dataset, hyperparamètres.
Centralisés ici pour garantir la reproductibilité des expériences.
"""

# ── Reproductibilité ───────────────────────────────────────────────────────────
GLOBAL_SEED = 42

# ── Dataset ────────────────────────────────────────────────────────────────────
DATASET = {
    "n_train":       300,
    "sigma_noise":   0.01,
    "n_test":        1000,
    "n_quadrature":  50_000,   # points pour l'intégration numérique
}

# ── Monte-Carlo ────────────────────────────────────────────────────────────────
MONTE_CARLO = {
    "n_samples_list": [100, 500, 1_000, 5_000, 10_000, 50_000, 100_000, 500_000],
    "n_bootstrap":    2_000,   # répétitions pour valider le TCL
    "n_fixed_tcl":    1_000,   # taille fixe pour illustration TCL
}

# ── Régression Polynomiale ─────────────────────────────────────────────────────
POLYNOMIAL = {
    "max_degree":  15,
    "alpha_ridge": 1e-12,
    "cv_folds":    5,
    "degrees_range": list(range(2, 16)),
}

# ── MLP ────────────────────────────────────────────────────────────────────────
MLP = {
    "layer_sizes": [1, 64, 64, 32, 1],
    "n_epochs":    4000,
    "batch_size":  64,
    "lr":          5e-4,
    "beta1":       0.9,
    "beta2":       0.999,
    "eps_adam":    1e-8,
    "verbose_every": 1000,
}

# ── Processus Gaussien ─────────────────────────────────────────────────────────
GAUSSIAN_PROCESS = {
    "sigma_f":       1.0,
    "length_scale":  0.2,
    "sigma_n":       0.01,
    "n_restarts":    8,
    "optimize":      True,
    "n_gp_max":      200,      # max points pour rester en O(n³) raisonnable
    "n_mc_integral": 10_000,
}

# ── Comparaison ────────────────────────────────────────────────────────────────
COMPARISON = {
    "ns_list":   [50, 100, 200, 400, 700, 1000, 2000, 5000],
    "n_seeds":   5,
    "delta_pac": 0.05,
}

# ── Rademacher ─────────────────────────────────────────────────────────────────
RADEMACHER = {
    "n_mc_rad": 500,
    "degrees":  list(range(1, 14)),
    "ns_list":  [50, 200, 500],
}

# ── Paths ──────────────────────────────────────────────────────────────────────
import os
BASE_DIR     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR  = os.path.join(BASE_DIR, "results")
FIGURES_DIR  = os.path.join(RESULTS_DIR, "figures")
DATA_DIR     = os.path.join(RESULTS_DIR, "data")
