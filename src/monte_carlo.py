"""
src/monte_carlo.py
──────────────────
Estimateur de π par la méthode de Monte-Carlo stochastique.

Théorie :
    Soit X = 1_{U₁²+U₂²≤1} avec (U₁,U₂) ~ U([0,1]²).
    E[X] = π/4  ⟹  π̂_n = 4/n Σᵢ 1_{U₁ᵢ²+U₂ᵢ²≤1}

Convergence (TCL) :
    √n (π̂_n - π) → N(0, π(4-π))  [σ² ≈ 2.654]
    RMSE = O(n^{-1/2})

Références :
    - Metropolis & Ulam (1949), "The Monte Carlo Method"
    - Sobol (1967), quasi-Monte Carlo (extension)
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Optional
from scipy.stats import norm as scipy_norm

PI_REF = np.pi


@dataclass
class MonteCarloResult:
    """Résultat complet d'une estimation Monte-Carlo de π."""
    n:          int
    pi_hat:     float
    std_error:  float
    ci_low:     float
    ci_high:    float
    error_abs:  float = field(init=False)
    inside:     Optional[np.ndarray] = field(default=None, repr=False)
    points:     Optional[np.ndarray] = field(default=None, repr=False)

    def __post_init__(self):
        self.error_abs = abs(self.pi_hat - PI_REF)

    def __str__(self) -> str:
        return (
            f"Monte-Carlo (n={self.n:,})\n"
            f"  π̂  = {self.pi_hat:.10f}\n"
            f"  err = {self.error_abs:.3e}\n"
            f"  IC95% = [{self.ci_low:.6f}, {self.ci_high:.6f}]"
        )


class MonteCarloEstimator:
    """
    Estimateur de π par la méthode de Monte-Carlo.

    Implémentation vectorisée avec NumPy, avec :
      - Estimation simple
      - Analyse de convergence multi-taille
      - Validation du Théorème Central Limite
      - Variance de contrôle optionnelle (réduction de variance)
    """

    def __init__(self, seed: int = 42):
        self.seed = seed
        self._rng  = np.random.default_rng(seed)

    def estimate(
        self,
        n: int,
        seed: Optional[int] = None,
        store_points: bool = False,
    ) -> MonteCarloResult:
        """
        Estime π avec n points i.i.d. ~ U([0,1]²).

        Paramètres
        ----------
        n            : nombre de tirages
        seed         : graine (None = utilise l'état courant du générateur)
        store_points : si True, stocke les coordonnées des points

        Retourne
        --------
        MonteCarloResult avec π̂, IC 95%, erreur, etc.
        """
        rng = np.random.default_rng(seed) if seed is not None else self._rng

        pts    = rng.uniform(0.0, 1.0, (n, 2))
        inside = (pts[:, 0]**2 + pts[:, 1]**2) <= 1.0

        p_hat   = inside.mean()
        pi_hat  = 4.0 * p_hat
        std_err = 4.0 * np.sqrt(p_hat * (1.0 - p_hat) / n)
        ci_low  = pi_hat - 1.96 * std_err
        ci_high = pi_hat + 1.96 * std_err

        return MonteCarloResult(
            n=n,
            pi_hat=pi_hat,
            std_error=std_err,
            ci_low=ci_low,
            ci_high=ci_high,
            inside=inside if store_points else None,
            points=pts if store_points else None,
        )

    def convergence_analysis(
        self,
        ns: list[int],
        seed: int = 0,
    ) -> dict:
        """
        Analyse la convergence de l'estimateur sur plusieurs tailles n.

        Retourne
        --------
        dict avec keys : 'ns', 'pi_hats', 'errors', 'std_errors'
        """
        pi_hats, errors, stds = [], [], []
        for n in ns:
            res = self.estimate(n, seed=seed)
            pi_hats.append(res.pi_hat)
            errors.append(res.error_abs)
            stds.append(res.std_error)

        return {
            "ns":         np.array(ns),
            "pi_hats":    np.array(pi_hats),
            "errors":     np.array(errors),
            "std_errors": np.array(stds),
        }

    def validate_tcl(
        self,
        n_fixed: int = 1000,
        n_simulations: int = 2000,
    ) -> dict:
        """
        Valide le Théorème Central Limite par simulation :
            √n (π̂_n - π) → N(0, π(4-π))

        Retourne
        --------
        dict : samples, sigma_theoretical, mean_sample, std_sample
        """
        samples = np.array([
            self.estimate(n_fixed, seed=s).pi_hat
            for s in range(n_simulations)
        ])
        sigma_th = np.sqrt(PI_REF * (4 - PI_REF) / n_fixed)
        return {
            "samples":           samples,
            "n_fixed":           n_fixed,
            "sigma_theoretical": sigma_th,
            "mean_sample":       samples.mean(),
            "std_sample":        samples.std(),
            "kolmogorov_smirnov": self._ks_test(samples, PI_REF, sigma_th),
        }

    def control_variate_estimate(self, n: int, seed: int = 0) -> tuple[float, float]:
        """
        Estimateur avec variable de contrôle pour réduction de variance.

        Variable de contrôle : g(x,y) = x + y  (E[g] = 1 connu)
        Réduit la variance d'un facteur typique de ~30%.

        Retourne : (π̂_cv, variance_réduite)
        """
        rng    = np.random.default_rng(seed)
        pts    = rng.uniform(0.0, 1.0, (n, 2))
        f_vals = (pts[:, 0]**2 + pts[:, 1]**2 <= 1.0).astype(float)
        g_vals = pts[:, 0] + pts[:, 1]  # E[g] = 1

        # Coefficient optimal de contrôle
        cov_fg = np.cov(f_vals, g_vals)[0, 1]
        var_g  = np.var(g_vals)
        c_star = -cov_fg / var_g if var_g > 0 else 0.0

        # Estimateur corrigé
        h_vals = f_vals + c_star * (g_vals - 1.0)
        pi_cv  = 4.0 * h_vals.mean()
        var_cv = 4.0**2 * h_vals.var() / n

        return pi_cv, var_cv

    @staticmethod
    def _ks_test(samples: np.ndarray, mu: float, sigma: float) -> dict:
        """Test KS pour vérifier la normalité asymptotique."""
        from scipy.stats import kstest
        stat, pval = kstest(samples, "norm", args=(mu, sigma))
        return {"statistic": stat, "p_value": pval}

    @staticmethod
    def theoretical_variance(n: int) -> float:
        """Variance théorique de l'estimateur : π(4-π)/n."""
        return PI_REF * (4 - PI_REF) / n

    @staticmethod
    def theoretical_rmse(n: int) -> float:
        """RMSE théorique : √(π(4-π)/n)."""
        return np.sqrt(MonteCarloEstimator.theoretical_variance(n))


if __name__ == "__main__":
    mc = MonteCarloEstimator(seed=42)

    print("=" * 55)
    print("  ESTIMATEUR MONTE-CARLO DE π")
    print("=" * 55)

    for n in [1_000, 10_000, 100_000]:
        res = mc.estimate(n)
        print(res)
        print()

    # Validation TCL
    tcl = mc.validate_tcl(n_fixed=1000, n_simulations=2000)
    print(f"Validation TCL :")
    print(f"  σ théorique  = {tcl['sigma_theoretical']:.6f}")
    print(f"  σ empirique  = {tcl['std_sample']:.6f}")
    print(f"  KS p-value   = {tcl['kolmogorov_smirnov']['p_value']:.4f}")
    print(f"  (p > 0.05 ⟹ H₀ : normalité non rejetée)")

    # Variable de contrôle
    pi_cv, var_cv = mc.control_variate_estimate(100_000)
    print(f"\nVariante variable de contrôle (n=100k) :")
    print(f"  π̂_cv = {pi_cv:.8f}  |  err = {abs(pi_cv - PI_REF):.2e}")
