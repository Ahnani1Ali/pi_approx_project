"""
src/rademacher.py
─────────────────
Calcul de la complexité de Rademacher empirique et des bornes PAC.

Théorie (Bartlett & Mendelson, JMLR 2002) :
    Avec probabilité ≥ 1-δ, ∀f ∈ H :
    R(f) ≤ R̂_n(f) + 2·𝔑̂_n(H) + √(log(1/δ)/(2n))

Complexité de Rademacher empirique :
    𝔑̂_n(H) = E_σ[sup_{f∈H} (1/n) Σᵢ σᵢ f(xᵢ)]
    σᵢ ~ Rademacher(1/2), i.i.d.

Pour la classe polynomiale normalisée Pd :
    sup est atteint analytiquement : 𝔑̂_n = E[||Φᵀσ||₂ / n]

Référence :
    Bartlett, P. L., & Mendelson, S. (2002). JMLR 3, 463–482.
"""

from __future__ import annotations

import numpy as np
from typing import Optional
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline

PI_REF = np.pi


def rademacher_polynomial(
    X: np.ndarray,
    degree: int,
    n_mc: int = 500,
    seed: int = 0,
) -> float:
    """
    Estime 𝔑̂_n(P_d) par simulation Monte-Carlo.

    Pour la classe polynomiale à norme bornée :
        sup_{||θ||₂ ≤ 1} (1/n) σᵀ Φ θ = (1/n) ||Φᵀ σ||₂

    Paramètres
    ----------
    X      : abscisses d'entraînement (n points)
    degree : degré polynomial d
    n_mc   : nombre de tirages de σ
    seed   : graine aléatoire

    Retourne
    --------
    float : estimation de 𝔑̂_n(P_d)
    """
    rng  = np.random.default_rng(seed)
    n    = len(X)
    Phi  = np.column_stack([X**k for k in range(degree + 1)])  # (n, d+1)

    sups = np.array([
        np.linalg.norm(Phi.T @ rng.choice([-1.0, 1.0], size=n)) / n
        for _ in range(n_mc)
    ])
    return float(sups.mean())


def pac_bound(
    X: np.ndarray,
    y: np.ndarray,
    degree: int,
    alpha_ridge: float = 1e-12,
    delta: float = 0.05,
    n_mc_rad: int = 200,
) -> dict:
    """
    Calcule la borne PAC complète pour un modèle polynomial.

    Borne : R(f) ≤ R̂_n(f) + 2𝔑̂_n(H) + √(log(1/δ)/(2n))

    Paramètres
    ----------
    X, y        : données d'entraînement
    degree      : degré polynomial
    alpha_ridge : régularisation Ridge
    delta       : niveau de confiance (probabilité d'échec ≤ δ)
    n_mc_rad    : tirages MC pour Rademacher

    Retourne
    --------
    dict : emp_risk, rad_complexity, conf_term, pac_bound
    """
    n = len(X)

    # Risque empirique (MSE sur les données d'entraînement)
    pipe = Pipeline([
        ("poly",  PolynomialFeatures(degree=degree, include_bias=True)),
        ("ridge", Ridge(alpha=alpha_ridge, fit_intercept=False)),
    ])
    pipe.fit(X.reshape(-1, 1), y)
    y_pred   = pipe.predict(X.reshape(-1, 1))
    emp_risk = float(np.mean((y_pred - y)**2))

    # Complexité de Rademacher
    rad = rademacher_polynomial(X, degree, n_mc=n_mc_rad)

    # Terme de confiance (Hoeffding)
    conf_term = float(np.sqrt(np.log(1.0 / delta) / (2.0 * n)))

    return {
        "n":          n,
        "degree":     degree,
        "emp":        emp_risk,
        "rad":        rad,
        "conf":       conf_term,
        "bound":      emp_risk + 2.0 * rad + conf_term,
    }


def compute_rademacher_vs_degree(
    X: np.ndarray,
    degrees: list[int],
    n_mc: int = 500,
    seed: int = 0,
) -> dict:
    """
    Calcule 𝔑̂_n(P_d) pour chaque degré dans la liste.

    Retourne
    --------
    dict : degree → complexité
    """
    return {
        d: rademacher_polynomial(X, d, n_mc=n_mc, seed=seed)
        for d in degrees
    }


def compute_pac_vs_n(
    ns: list[int],
    sigma_noise: float = 0.01,
    degree: int = 7,
    delta: float = 0.05,
    n_mc_rad: int = 200,
    seed: int = 42,
) -> list[dict]:
    """
    Calcule la décomposition PAC pour plusieurs tailles de dataset.

    Retourne
    --------
    Liste de dict (un par n) avec emp, rad, conf, bound
    """
    from src.dataset import generate_dataset
    results = []
    for n in ns:
        ds = generate_dataset(n=n, sigma_noise=sigma_noise, seed=seed)
        result = pac_bound(
            ds.X_train, ds.y_train,
            degree=degree, delta=delta, n_mc_rad=n_mc_rad
        )
        results.append(result)
    return results


if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from src.dataset import generate_dataset

    ds = generate_dataset(n=200, sigma_noise=0.01, seed=42)

    print("Complexité de Rademacher 𝔑̂_n(P_d) :")
    for d in [2, 4, 6, 8, 10]:
        r = rademacher_polynomial(ds.X_train, d, n_mc=500)
        print(f"  d={d:2d}  𝔑̂_n = {r:.5f}")

    print("\nBorne PAC (d=7, δ=0.05) :")
    result = pac_bound(ds.X_train, ds.y_train, degree=7)
    for k, v in result.items():
        if isinstance(v, float):
            print(f"  {k:<15} : {v:.4e}")
