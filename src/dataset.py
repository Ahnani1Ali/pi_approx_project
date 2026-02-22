"""
src/dataset.py
──────────────
Génération du jeu de données synthétique pour l'approximation de π.

Principe géométrique :
    ∫₀¹ √(1-x²) dx = π/4

On observe donc la fonction f(x) = √(1-x²) bruitée :
    y_i = √(1-x_i²) + ε_i,  ε_i ~ N(0, σ²)

Auteur : M1 Informatique — IA
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Optional


# ── Valeur de référence ────────────────────────────────────────────────────────
PI_REF = np.pi


@dataclass
class Dataset:
    """
    Conteneur pour le jeu de données synthétique.

    Attributs
    ---------
    X_train    : abscisses d'entraînement (triées, dans [0,1])
    y_train    : ordonnées bruitées
    y_clean    : ordonnées sans bruit (f(x) exacte)
    X_test     : grille de test dense
    y_test     : valeurs exactes sur la grille de test
    sigma_noise: écart-type du bruit gaussien utilisé
    n          : nombre de points d'entraînement
    """
    X_train:     np.ndarray
    y_train:     np.ndarray
    y_clean:     np.ndarray
    X_test:      np.ndarray
    y_test:      np.ndarray
    sigma_noise: float
    n:           int = field(init=False)

    def __post_init__(self):
        self.n = len(self.X_train)

    def snr(self) -> float:
        """Signal-to-Noise Ratio en dB."""
        signal_power = np.mean(self.y_clean ** 2)
        noise_power  = self.sigma_noise ** 2
        return 10 * np.log10(signal_power / noise_power)

    def summary(self) -> str:
        lines = [
            "═" * 50,
            "  DATASET SUMMARY",
            "═" * 50,
            f"  n_train     : {self.n}",
            f"  n_test      : {len(self.X_test)}",
            f"  σ_noise     : {self.sigma_noise:.4f}",
            f"  SNR         : {self.snr():.1f} dB",
            f"  x ∈         : [{self.X_train.min():.3f}, {self.X_train.max():.3f}]",
            f"  y ∈         : [{self.y_clean.min():.3f}, {self.y_clean.max():.3f}]",
            "═" * 50,
        ]
        return "\n".join(lines)


def target_function(x: np.ndarray) -> np.ndarray:
    """
    Fonction cible : f(x) = √(1 - x²) — quart de cercle supérieur.

    Propriété fondamentale :
        4 * ∫₀¹ f(x) dx = π

    Paramètres
    ----------
    x : array de valeurs dans [0, 1]

    Retourne
    --------
    f(x) = √(1 - x²)
    """
    x = np.asarray(x, dtype=float)
    # Clip pour éviter des NaN à x=1 à cause d'erreurs numériques
    return np.sqrt(np.clip(1 - x**2, 0, None))


def generate_dataset(
    n: int,
    sigma_noise: float = 0.01,
    n_test: int = 1000,
    seed: int = 42,
) -> Dataset:
    """
    Génère le dataset synthétique basé sur le quart de cercle.

    Paramètres
    ----------
    n           : nombre de points d'entraînement
    sigma_noise : écart-type du bruit gaussien N(0, σ²)
    n_test      : nombre de points sur la grille de test
    seed        : graine aléatoire (reproductibilité)

    Retourne
    --------
    Dataset : objet contenant X_train, y_train, y_clean, X_test, y_test
    """
    rng = np.random.default_rng(seed)

    # Points d'entraînement : tirés uniformément dans [0,1]
    X_train = np.sort(rng.uniform(0.0, 1.0, n))
    y_clean = target_function(X_train)
    y_train = y_clean + rng.normal(0.0, sigma_noise, n)

    # Grille de test dense (équidistante)
    X_test = np.linspace(0.0, 1.0, n_test)
    y_test = target_function(X_test)

    return Dataset(
        X_train=X_train,
        y_train=y_train,
        y_clean=y_clean,
        X_test=X_test,
        y_test=y_test,
        sigma_noise=sigma_noise,
    )


def monte_carlo_unit_square(n: int, seed: int = 0) -> tuple[float, float, np.ndarray]:
    """
    Génère n points i.i.d. uniformes dans [0,1]² pour Monte-Carlo.

    Retourne
    --------
    (pi_hat, std_error, points_array)
        pi_hat      : estimation de π
        std_error   : écart-type de l'estimateur (IC 95% = ±1.96*std_error)
        points      : array (n, 2) des points tirés
    """
    rng    = np.random.default_rng(seed)
    points = rng.uniform(0.0, 1.0, (n, 2))
    inside = (points[:, 0]**2 + points[:, 1]**2) <= 1.0

    p_hat   = inside.mean()
    pi_hat  = 4.0 * p_hat
    std_err = 4.0 * np.sqrt(p_hat * (1.0 - p_hat) / n)

    return pi_hat, std_err, points


if __name__ == "__main__":
    ds = generate_dataset(n=300, sigma_noise=0.01, seed=42)
    print(ds.summary())
    print(f"\nTest : 4 * ∫f(x)dx (exact) = {4 * np.trapz(ds.y_test, ds.X_test):.8f}")
    print(f"Valeur exacte de π         = {PI_REF:.8f}")
