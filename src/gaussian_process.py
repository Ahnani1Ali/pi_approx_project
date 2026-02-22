"""
src/gaussian_process.py
───────────────────────
Régression par Processus Gaussien avec noyau RBF.
Implémentation NumPy avec factorisation de Cholesky et MLE-II.

Modèle :
    f ~ GP(0, k_RBF)
    y = f(x) + ε,  ε ~ N(0, σ_n²)

Noyau RBF :
    k(x,x') = σ_f² exp(-||x-x'||²/(2ℓ²))

Postérieure exacte :
    μ*(x*) = k*ᵀ C⁻¹ y                [Eq. 2.23, R&W 2006]
    σ*²(x*) = k(x*,x*) - k*ᵀ C⁻¹ k*  [Eq. 2.24]
    C = K(X,X) + σ_n² I

Hyperparamètres : optimisés par MLE-II (log-vraisemblance marginale)
    log p(y|X,θ) = -½ yᵀC⁻¹y - ½ log|C| - n/2 log(2π)

Références :
    - Rasmussen & Williams (2006), "GP for Machine Learning", MIT Press
    - Snelson & Ghahramani (2006), Sparse GP (NeurIPS)
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Tuple
from scipy.optimize import minimize
from scipy.linalg import cho_factor, cho_solve

PI_REF = np.pi


@dataclass
class GPResult:
    """Résultat de l'estimation de π par processus gaussien."""
    pi_hat:      float
    pi_std:      float
    sigma_f:     float
    length_scale: float
    sigma_n:     float
    log_mlik:    float
    error_abs:   float = field(init=False)

    def __post_init__(self):
        self.error_abs = abs(self.pi_hat - PI_REF)

    def __str__(self) -> str:
        return (
            f"Processus Gaussien (noyau RBF)\n"
            f"  Hyperparamètres optimisés :\n"
            f"    σ_f = {self.sigma_f:.5f}  |  ℓ = {self.length_scale:.5f}  |  σ_n = {self.sigma_n:.5f}\n"
            f"  Log-vraisemblance marginale = {self.log_mlik:.4f}\n"
            f"  π̂  = {self.pi_hat:.10f}\n"
            f"  σ_π = {self.pi_std:.3e}\n"
            f"  IC bayésien ±2σ = [{self.pi_hat - 2*self.pi_std:.8f}, "
            f"{self.pi_hat + 2*self.pi_std:.8f}]\n"
            f"  err = {self.error_abs:.3e}"
        )


class GaussianProcessRegressor:
    """
    Régression par Processus Gaussien — noyau RBF.

    Paramètres
    ----------
    sigma_f      : amplitude du signal (σ_f > 0)
    length_scale : échelle de longueur ℓ (> 0)
    sigma_n      : bruit d'observation σ_n (> 0)
    jitter       : terme de jitter numérique pour la stabilité Cholesky

    Attributs publics après fit()
    -----------------------------
    X_train_, y_train_  : données d'entraînement
    L_chol_             : facteur de Cholesky de C = K + σ_n²I
    alpha_              : vecteur C⁻¹ y (pour la prédiction)
    result_             : GPResult complet
    """

    def __init__(
        self,
        sigma_f: float       = 1.0,
        length_scale: float  = 0.3,
        sigma_n: float       = 0.01,
        jitter: float        = 1e-6,
    ):
        self._log_params = np.log([sigma_f, length_scale, sigma_n])
        self.jitter      = jitter
        self.X_train_    = None
        self.y_train_    = None
        self.L_chol_     = None
        self.alpha_      = None
        self.result_     = None

    # ── Propriétés (via log-space pour garantir la positivité) ────────────────
    @property
    def sigma_f(self) -> float:
        return float(np.exp(self._log_params[0]))

    @property
    def length_scale(self) -> float:
        return float(np.exp(self._log_params[1]))

    @property
    def sigma_n(self) -> float:
        return float(np.exp(self._log_params[2]))

    # ── Noyaux ────────────────────────────────────────────────────────────────
    def rbf_kernel(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """
        Noyau RBF (Squared Exponential / Gaussian) :
            k(x,x') = σ_f² exp(-||x-x'||²/(2ℓ²))
        """
        X1 = X1.reshape(-1, 1)
        X2 = X2.reshape(-1, 1)
        sq_dists = (X1 - X2.T)**2
        return self.sigma_f**2 * np.exp(-sq_dists / (2.0 * self.length_scale**2))

    def matern52_kernel(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """
        Noyau de Matérn ν=5/2 (alternative au RBF) :
            k(r) = σ_f²(1 + √5 r/ℓ + 5r²/(3ℓ²)) exp(-√5 r/ℓ)
        """
        X1 = X1.reshape(-1, 1)
        X2 = X2.reshape(-1, 1)
        r   = np.abs(X1 - X2.T)
        sq5 = np.sqrt(5.0)
        z   = sq5 * r / self.length_scale
        return self.sigma_f**2 * (1.0 + z + z**2 / 3.0) * np.exp(-z)

    def _build_covariance(self, X: np.ndarray) -> np.ndarray:
        """Matrice de covariance C = K(X,X) + (σ_n² + jitter) I."""
        n = len(X)
        K = self.rbf_kernel(X, X)
        return K + (self.sigma_n**2 + self.jitter) * np.eye(n)

    # ── Log-vraisemblance marginale ────────────────────────────────────────────
    def neg_log_marginal_likelihood(self, log_params: np.ndarray) -> float:
        """
        Log-vraisemblance marginale négative (objective MLE-II) :
            -log p(y|X,θ) = ½ yᵀC⁻¹y + ½ log|C| + n/2 log(2π)

        Paramètre
        ---------
        log_params : [log(σ_f), log(ℓ), log(σ_n)]
        """
        self._log_params = log_params
        n = len(self.y_train_)
        try:
            C = self._build_covariance(self.X_train_)
            L = np.linalg.cholesky(C)
        except np.linalg.LinAlgError:
            return 1e10

        alpha = np.linalg.solve(L.T, np.linalg.solve(L, self.y_train_))
        # Data fit term + complexity penalty (rasoir d'Occam)
        nll = (
            0.5 * self.y_train_ @ alpha
            + np.sum(np.log(np.diag(L)))
            + 0.5 * n * np.log(2.0 * np.pi)
        )
        return float(nll)

    def log_marginal_likelihood(self) -> float:
        """Retourne la log-vraisemblance marginale (en norme positive)."""
        return -self.neg_log_marginal_likelihood(self._log_params)

    # ── Entraînement ──────────────────────────────────────────────────────────
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        optimize: bool = True,
        n_restarts: int = 8,
        bounds: Optional[list] = None,
    ) -> 'GaussianProcessRegressor':
        """
        Ajuste le GP aux données par optimisation des hyperparamètres (MLE-II).

        Stratégie d'optimisation : L-BFGS-B avec n_restarts initialisations
        aléatoires pour éviter les minima locaux de la vraisemblance.

        Paramètres
        ----------
        X, y       : données d'entraînement
        optimize   : si False, garde les hyperparamètres courants
        n_restarts : nombre de redémarrages aléatoires
        bounds     : bornes en log-space [(min,max) × 3]
        """
        self.X_train_ = X.copy()
        self.y_train_ = y.copy()

        if optimize:
            if bounds is None:
                bounds = [(-4.0, 2.0), (-5.0, 1.0), (-6.0, 0.0)]

            best_nll    = np.inf
            best_params = self._log_params.copy()
            rng = np.random.default_rng(42)

            for _ in range(n_restarts):
                x0 = np.array([
                    rng.uniform(b[0], b[1]) for b in bounds
                ])
                try:
                    result = minimize(
                        self.neg_log_marginal_likelihood,
                        x0,
                        method="L-BFGS-B",
                        bounds=bounds,
                        options={"maxiter": 200, "ftol": 1e-12},
                    )
                    if result.fun < best_nll:
                        best_nll    = result.fun
                        best_params = result.x.copy()
                except Exception:
                    continue

            self._log_params = best_params

        # Factorisation de Cholesky finale (précalcul)
        C = self._build_covariance(X)
        self.L_chol_ = cho_factor(C, lower=True)
        self.alpha_  = cho_solve(self.L_chol_, y)

        return self

    # ── Prédiction ────────────────────────────────────────────────────────────
    def predict(
        self,
        X_star: np.ndarray,
        return_std: bool = True,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Calcule la moyenne et l'écart-type de la postérieure GP.

        μ*(x*) = K(x*,X) C⁻¹ y         = k*ᵀ α
        σ*²(x*) = k(x*,x*) - K(x*,X) C⁻¹ K(X,x*)

        Retourne
        --------
        (mu, sigma) si return_std=True, sinon (mu, None)
        """
        if self.alpha_ is None:
            raise RuntimeError("Appeler fit() avant predict().")

        k_star = self.rbf_kernel(X_star, self.X_train_)  # (M, n)
        mu     = k_star @ self.alpha_

        if not return_std:
            return mu, None

        v    = cho_solve(self.L_chol_, k_star.T)          # (n, M)
        var  = self.sigma_f**2 - np.sum(k_star.T * v, axis=0)
        var  = np.maximum(var, 1e-12)                      # clip numérique
        return mu, np.sqrt(var)

    def sample_posterior(
        self,
        X_star: np.ndarray,
        n_samples: int = 5,
        seed: int = 0,
    ) -> np.ndarray:
        """
        Génère des trajectoires de la distribution postérieure.

        Retourne
        --------
        array (n_samples, len(X_star))
        """
        mu, std = self.predict(X_star)
        m = len(X_star)
        # Covariance postérieure complète
        k_star = self.rbf_kernel(X_star, self.X_train_)
        K_star = self.rbf_kernel(X_star, X_star)
        v      = cho_solve(self.L_chol_, k_star.T)
        Sigma  = K_star - k_star @ v
        Sigma  += 1e-8 * np.eye(m)  # jitter

        rng = np.random.default_rng(seed)
        L   = np.linalg.cholesky(Sigma)
        return mu + (L @ rng.normal(size=(m, n_samples))).T

    # ── Estimation de π ───────────────────────────────────────────────────────
    def estimate_pi(self, M: int = 10_000) -> Tuple[float, float]:
        """
        Estime π et son incertitude bayésienne par intégration de la postérieure.

        π̂    = 4 ∫₀¹ μ*(x) dx  ≈ 4 * trapz(μ*(x_grid))
        σ_π  ≈ 4 √(∫₀¹ σ*²(x) dx)  [borne conservatrice]

        Retourne
        --------
        (pi_hat, pi_std)
        """
        x_quad  = np.linspace(0.0, 1.0, M)
        mu, std = self.predict(x_quad)

        dx     = x_quad[1] - x_quad[0]
        pi_hat = 4.0 * np.trapezoid(mu, x_quad)
        pi_std = 4.0 * np.sqrt(np.sum(std**2) * dx)

        return pi_hat, pi_std

    def fit_and_estimate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        **fit_kwargs,
    ) -> GPResult:
        """Raccourci : ajuste et retourne un GPResult complet."""
        self.fit(X, y, **fit_kwargs)
        pi_hat, pi_std = self.estimate_pi()
        log_mlik = self.log_marginal_likelihood()

        self.result_ = GPResult(
            pi_hat=pi_hat,
            pi_std=pi_std,
            sigma_f=self.sigma_f,
            length_scale=self.length_scale,
            sigma_n=self.sigma_n,
            log_mlik=log_mlik,
        )
        return self.result_


if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from src.dataset import generate_dataset

    ds  = generate_dataset(n=200, sigma_noise=0.01, seed=42)

    print("Entraînement du Processus Gaussien (MLE-II)...")
    gp  = GaussianProcessRegressor(sigma_f=1.0, length_scale=0.2, sigma_n=0.01)
    res = gp.fit_and_estimate(
        ds.X_train[:200], ds.y_train[:200],
        optimize=True, n_restarts=8
    )
    print(res)

    # Intervalles de confiance bayésiens
    mu, std = gp.predict(np.array([0.0, 0.25, 0.5, 0.75, 1.0]))
    print("\nPrédictions aux points de test :")
    x_pts = [0.0, 0.25, 0.5, 0.75, 1.0]
    for x, m, s in zip(x_pts, mu, std):
        print(f"  f({x:.2f}) ≈ {m:.6f} ± {2*s:.6f}  [IC 95%]")
