"""
src/__init__.py
───────────────
Package d'approximation de π par machine learning.

Modules disponibles :
    dataset             — Génération des données synthétiques
    monte_carlo         — Estimateur Monte-Carlo
    polynomial_regression — Régression polynomiale Ridge
    neural_network      — MLP NumPy pur (Adam, GELU, rétropropagation)
    gaussian_process    — GP avec noyau RBF et MLE-II
    rademacher          — Complexité de Rademacher et bornes PAC
    visualization       — Figures publication-quality
"""

from src.dataset import generate_dataset, Dataset, target_function, PI_REF
from src.monte_carlo import MonteCarloEstimator
from src.polynomial_regression import PolynomialPiEstimator
from src.neural_network import MLP
from src.gaussian_process import GaussianProcessRegressor

__version__ = "1.0.0"
__author__  = "M1 Informatique — IA"
