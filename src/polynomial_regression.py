"""
src/polynomial_regression.py
─────────────────────────────
Régression polynomiale Ridge pour l'approximation de f(x)=√(1-x²)
et l'estimation de π par intégration analytique exacte.

Cadre mathématique :
    Classe de fonctions : P_d = {p : p(x) = Σ_{k=0}^d θ_k x^k}
    Estimateur Ridge    : θ̂ = (Φᵀ Φ + λI)⁻¹ Φᵀ y
    Intégration exacte  : π̂ = 4 ∫₀¹ p(x)dx = 4 Σ_k θ_k/(k+1)
    Sélection de degré  : validation croisée K-fold (MSE)

Références :
    - Trefethen (2019), "Approximation Theory and Approximation Practice"
    - Bartlett & Mendelson (2002), complexité de Rademacher
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Optional
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, KFold

PI_REF = np.pi


@dataclass
class PolynomialResult:
    """Résultat de la régression polynomiale."""
    pi_hat:      float
    best_degree: int
    theta:       np.ndarray
    cv_scores:   dict
    error_abs:   float = field(init=False)
    integral_exact: float = field(init=False)

    def __post_init__(self):
        self.error_abs     = abs(self.pi_hat - PI_REF)
        self.integral_exact = self.pi_hat / 4.0

    def __str__(self) -> str:
        return (
            f"Régression Polynomiale (d={self.best_degree})\n"
            f"  π̂  = {self.pi_hat:.10f}\n"
            f"  err = {self.error_abs:.3e}\n"
            f"  ∫₀¹ p(x)dx ≈ {self.integral_exact:.10f}  (exact = {PI_REF/4:.10f})"
        )


class PolynomialPiEstimator:
    """
    Estimateur de π par régression polynomiale Ridge avec sélection de degré.

    Algorithme :
    1. Sélectionner le degré d optimal par CV-K (minimisation MSE)
    2. Ajuster p_d(x) = Σ θ_k x^k sur tout le jeu d'entraînement
    3. Calculer π̂ = 4 Σ θ_k/(k+1)  [intégrale exacte de p_d sur [0,1]]

    Propriétés :
    - θ̂_OLS est sans biais (Gauss-Markov)
    - Ridge régularise l'inversion mal conditionnée de la matrice de Vandermonde
    - La convergence polynomiale sur f analytique est exponentielle en d
    """

    def __init__(
        self,
        max_degree: int = 15,
        alpha_ridge: float = 1e-12,
        cv_folds: int = 5,
        seed: int = 42,
    ):
        self.max_degree  = max_degree
        self.alpha_ridge = alpha_ridge
        self.cv_folds    = cv_folds
        self.seed        = seed
        self.pipeline_   = None
        self.result_     = None

    def _build_pipeline(self, degree: int) -> Pipeline:
        return Pipeline([
            ("poly",  PolynomialFeatures(degree=degree, include_bias=True)),
            ("ridge", Ridge(alpha=self.alpha_ridge, fit_intercept=False)),
        ])

    def select_degree(self, X: np.ndarray, y: np.ndarray) -> dict:
        """
        Sélectionne le degré optimal par validation croisée K-fold.

        Retourne
        --------
        dict : degree → MSE_cv moyen
        """
        kf = KFold(n_splits=self.cv_folds, shuffle=True, random_state=self.seed)
        cv_scores = {}
        for d in range(2, self.max_degree + 1):
            pipe   = self._build_pipeline(d)
            scores = cross_val_score(
                pipe, X.reshape(-1, 1), y,
                cv=kf, scoring="neg_mean_squared_error"
            )
            cv_scores[d] = -scores.mean()
        return cv_scores

    def fit(self, X: np.ndarray, y: np.ndarray) -> PolynomialResult:
        """
        Ajuste le modèle et calcule l'estimation de π.

        Étapes :
        (1) Sélection du degré par CV
        (2) Ajustement sur l'ensemble complet
        (3) Extraction des coefficients θ
        (4) Intégrale analytique : π̂ = 4 Σ θ_k/(k+1)

        Retourne
        --------
        PolynomialResult
        """
        cv_scores   = self.select_degree(X, y)
        best_degree = min(cv_scores, key=cv_scores.get)

        self.pipeline_ = self._build_pipeline(best_degree)
        self.pipeline_.fit(X.reshape(-1, 1), y)

        # Récupération des coefficients : θ_0, θ_1, ..., θ_d
        # Avec fit_intercept=False et include_bias=True dans PolynomialFeatures,
        # les coefficients incluent le biais (θ_0) directement dans coef_
        coef = self.pipeline_.named_steps["ridge"].coef_
        theta = coef.flatten()  # shape (d+1,) : [θ_0, θ_1, ..., θ_d]

        # Intégrale exacte : ∫₀¹ Σ θ_k x^k dx = Σ θ_k/(k+1)
        k_vals = np.arange(len(theta), dtype=float)
        pi_hat = 4.0 * float(np.sum(theta / (k_vals + 1.0)))

        self.result_ = PolynomialResult(
            pi_hat=pi_hat,
            best_degree=best_degree,
            theta=theta,
            cv_scores=cv_scores,
        )
        return self.result_

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Prédit les valeurs de f(x) pour un tableau X."""
        if self.pipeline_ is None:
            raise RuntimeError("Appeler fit() avant predict().")
        return self.pipeline_.predict(X.reshape(-1, 1))

    def gauss_legendre_integral(self, n_points: int = 100) -> float:
        """
        Intégration de Gauss-Legendre (alternative à l'intégrale analytique).
        Exacte pour les polynômes si n_points ≥ ⌈(d+1)/2⌉.

        Retourne : 4 * ∫₀¹ p̂(x) dx
        """
        nodes, weights = np.polynomial.legendre.leggauss(n_points)
        # Changement de variable : [−1,1] → [0,1]
        x_mapped = 0.5 * (nodes + 1.0)
        w_mapped = 0.5 * weights
        return 4.0 * np.sum(w_mapped * self.predict(x_mapped))

    def bias_variance_analysis(
        self,
        X: np.ndarray,
        y_true: np.ndarray,
        sigma_noise: float,
    ) -> dict:
        """
        Décomposition biais-variance pour chaque degré d.

        Retourne
        --------
        dict : degree → {mse_cv, bias_sq_approx, irreducible}
        """
        analysis = {}
        irreducible = sigma_noise**2
        for d, mse_cv in self.result_.cv_scores.items():
            analysis[d] = {
                "mse_cv":       mse_cv,
                "irreducible":  irreducible,
                "variance_est": max(0.0, mse_cv - irreducible),  # approx
            }
        return analysis

    def vandermonde_condition_number(self, X: np.ndarray) -> dict:
        """
        Calcule le conditionnement κ(ΦᵀΦ) pour chaque degré.
        Illustre la nécessité de la régularisation Ridge.
        """
        cond_numbers = {}
        for d in range(2, min(self.max_degree, 12) + 1):
            Phi = np.column_stack([X**k for k in range(d + 1)])
            G   = Phi.T @ Phi
            cond_numbers[d] = np.linalg.cond(G)
        return cond_numbers


if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from src.dataset import generate_dataset

    ds  = generate_dataset(n=300, sigma_noise=0.01, seed=42)
    est = PolynomialPiEstimator(max_degree=15, alpha_ridge=1e-12, cv_folds=5)
    res = est.fit(ds.X_train, ds.y_train)

    print(res)
    print(f"\nIntégrale Gauss-Legendre : π̂ = {est.gauss_legendre_integral():.10f}")

    # Conditionnement
    cond = est.vandermonde_condition_number(ds.X_train)
    print("\nConditionnement κ(ΦᵀΦ) :")
    for d, k in list(cond.items())[:8]:
        print(f"  d={d:2d}  κ = {k:.3e}")
