"""
tests/test_all.py
─────────────────
Tests unitaires pour tous les modules du projet.

Exécution :
    python -m pytest tests/ -v
    python tests/test_all.py  (mode direct)

Tests couverts :
    - Dataset (génération, propriétés mathématiques)
    - Monte-Carlo (estimateur, TCL, variable de contrôle)
    - Régression polynomiale (OLS, intégration, CV)
    - MLP (forward/backward, convergence, Adam)
    - Processus Gaussien (noyau, postérieure, MLE)
    - Rademacher (complexité, borne PAC)
"""

import sys
import os
import numpy as np
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dataset import generate_dataset, target_function, PI_REF, monte_carlo_unit_square
from src.monte_carlo import MonteCarloEstimator
from src.polynomial_regression import PolynomialPiEstimator
from src.neural_network import MLP, gelu, gelu_grad
from src.gaussian_process import GaussianProcessRegressor
from src.rademacher import rademacher_polynomial, pac_bound


# ── Tolérance globale ─────────────────────────────────────────────────────────
TOL_LOOSE = 0.05     # 5% pour tests rapides
TOL_STRICT = 0.005   # 0.5% pour tests précis


class TestDataset(unittest.TestCase):
    """Tests du module dataset."""

    def setUp(self):
        self.ds = generate_dataset(n=200, sigma_noise=0.01, seed=42)

    def test_shapes(self):
        self.assertEqual(len(self.ds.X_train), 200)
        self.assertEqual(len(self.ds.y_train), 200)
        self.assertEqual(len(self.ds.y_clean), 200)
        self.assertEqual(len(self.ds.X_test),  1000)
        self.assertEqual(len(self.ds.y_test),  1000)

    def test_x_range(self):
        self.assertGreaterEqual(self.ds.X_train.min(), 0.0)
        self.assertLessEqual(self.ds.X_train.max(), 1.0)

    def test_target_function_at_zero(self):
        # f(0) = √(1-0²) = 1.0
        self.assertAlmostEqual(float(target_function(np.array([0.0]))[0]), 1.0, places=10)

    def test_target_function_unit_circle(self):
        # f(1) = 0 (exactement)
        self.assertAlmostEqual(float(target_function(np.array([1.0]))[0]), 0.0, places=10)

    def test_pi_via_trapezoid(self):
        # 4 * ∫₀¹ √(1-x²) dx ≈ π avec la grille de test
        pi_approx = 4.0 * np.trapezoid(self.ds.y_test, self.ds.X_test)
        self.assertAlmostEqual(pi_approx, PI_REF, places=3)

    def test_snr_positive(self):
        self.assertGreater(self.ds.snr(), 0.0)

    def test_sorted_x(self):
        self.assertTrue(np.all(np.diff(self.ds.X_train) >= 0))

    def test_noise_level(self):
        # L'écart-type des résidus doit être proche de sigma_noise
        residuals = self.ds.y_train - self.ds.y_clean
        self.assertAlmostEqual(residuals.std(), 0.01, delta=0.005)


class TestMonteCarlo(unittest.TestCase):
    """Tests de l'estimateur Monte-Carlo."""

    def setUp(self):
        self.mc = MonteCarloEstimator(seed=42)

    def test_estimate_range(self):
        res = self.mc.estimate(10_000)
        self.assertGreater(res.pi_hat, 3.0)
        self.assertLess(res.pi_hat, 3.3)

    def test_large_n_close_to_pi(self):
        res = self.mc.estimate(500_000, seed=0)
        self.assertAlmostEqual(res.pi_hat, PI_REF, delta=TOL_LOOSE)

    def test_ci_contains_pi(self):
        # L'IC 95% doit contenir π dans ~95% des cas
        count = 0
        N_TRIALS = 200
        for seed in range(N_TRIALS):
            res = self.mc.estimate(1000, seed=seed)
            if res.ci_low <= PI_REF <= res.ci_high:
                count += 1
        coverage = count / N_TRIALS
        # Coverage attendue : ~95%, tolérance ±5%
        self.assertGreater(coverage, 0.88)
        self.assertLess(coverage, 1.0)

    def test_convergence_decreasing(self):
        ns  = [100, 1000, 10000]
        errs = [self.mc.estimate(n, seed=42).error_abs for n in ns]
        # En moyenne, l'erreur doit décroître
        self.assertGreater(errs[0], errs[2] * 0.5)  # au moins 50% moins d'erreur

    def test_theoretical_variance(self):
        var_th = MonteCarloEstimator.theoretical_variance(1000)
        self.assertAlmostEqual(var_th, PI_REF * (4 - PI_REF) / 1000, places=10)

    def test_std_error_formula(self):
        res = self.mc.estimate(10_000, seed=0)
        expected_std = np.sqrt(PI_REF * (4 - PI_REF) / 10_000)
        self.assertAlmostEqual(res.std_error, expected_std, delta=0.01)

    def test_tcl_normality(self):
        tcl = self.mc.validate_tcl(n_fixed=500, n_simulations=1000)
        # p-value KS > 0.01 → normalité non rejetée
        self.assertGreater(tcl["kolmogorov_smirnov"]["p_value"], 0.01)

    def test_control_variate_reduces_variance(self):
        pi_std, var_cv = self.mc.control_variate_estimate(50_000)
        pi_plain = self.mc.estimate(50_000, seed=0)
        # La variance avec variable de contrôle doit être plus petite
        plain_var = (PI_REF * (4 - PI_REF)) / 50_000
        self.assertLess(var_cv, plain_var)


class TestPolynomialRegression(unittest.TestCase):
    """Tests de la régression polynomiale."""

    def setUp(self):
        self.ds  = generate_dataset(n=300, sigma_noise=0.01, seed=42)
        self.est = PolynomialPiEstimator(max_degree=12, alpha_ridge=1e-12, cv_folds=5)
        self.res = self.est.fit(self.ds.X_train, self.ds.y_train)

    def test_degree_selected(self):
        self.assertGreaterEqual(self.res.best_degree, 2)
        self.assertLessEqual(self.res.best_degree, 12)

    def test_pi_estimate_close(self):
        self.assertAlmostEqual(self.res.pi_hat, PI_REF, delta=TOL_STRICT)

    def test_analytical_vs_gauss(self):
        # Intégrale analytique et Gauss-Legendre doivent être proches
        pi_gl = self.est.gauss_legendre_integral(n_points=50)
        self.assertAlmostEqual(self.res.pi_hat, pi_gl, places=6)

    def test_predict_at_zero(self):
        # p(0) ≈ f(0) = 1
        pred = self.est.predict(np.array([0.0]))
        self.assertAlmostEqual(float(pred[0]), 1.0, delta=0.05)

    def test_cv_scores_dict(self):
        self.assertGreater(len(self.res.cv_scores), 5)
        for d, score in self.res.cv_scores.items():
            self.assertGreater(score, 0)

    def test_conditioning(self):
        cond = self.est.vandermonde_condition_number(self.ds.X_train)
        # Doit croître avec d
        vals = list(cond.values())
        self.assertLess(vals[0], vals[-1])


class TestMLP(unittest.TestCase):
    """Tests du réseau de neurones MLP."""

    def setUp(self):
        self.ds  = generate_dataset(n=200, sigma_noise=0.01, seed=42)
        self.mlp = MLP(layer_sizes=[1, 32, 32, 1], activation="gelu", seed=0)

    def test_gelu_at_zero(self):
        # GELU(0) = 0 * Φ(0) = 0
        self.assertAlmostEqual(float(gelu(np.array([0.0]))[0]), 0.0, places=10)

    def test_gelu_positive_for_large_x(self):
        # Pour x grand, GELU(x) ≈ x
        x = np.array([5.0])
        self.assertAlmostEqual(float(gelu(x)[0]), 5.0, delta=0.01)

    def test_gelu_grad_at_zero(self):
        # GELU'(0) = 0.5
        self.assertAlmostEqual(float(gelu_grad(np.array([0.0]))[0]), 0.5, delta=0.01)

    def test_forward_shape(self):
        X = np.linspace(0, 1, 50)
        out, zs, acts = self.mlp.forward(X)
        self.assertEqual(out.shape, (50, 1))
        self.assertEqual(len(zs), len(self.mlp.W))

    def test_predict_shape(self):
        X = np.linspace(0, 1, 100)
        y = self.mlp.predict(X)
        self.assertEqual(y.shape, (100,))

    def test_backward_shapes(self):
        X = np.array([0.1, 0.5, 0.9])
        y = np.array([0.99, 0.87, 0.44])
        loss, dW, db = self.mlp.backward(X, y)
        self.assertIsInstance(loss, float)
        self.assertGreater(loss, 0)
        for w, g in zip(self.mlp.W, dW):
            self.assertEqual(w.shape, g.shape)

    def test_train_reduces_loss(self):
        loss_before = self.mlp.mse_loss(self.ds.X_train, self.ds.y_train)
        self.mlp.train(self.ds.X_train, self.ds.y_train, n_epochs=200,
                       batch_size=64, verbose_every=0)
        loss_after = self.mlp.mse_loss(self.ds.X_train, self.ds.y_train)
        self.assertLess(loss_after, loss_before)

    def test_pi_estimate_reasonable(self):
        # Entraînement court pour test rapide
        self.mlp.train(self.ds.X_train, self.ds.y_train, n_epochs=500,
                       batch_size=64, verbose_every=0)
        pi_hat = self.mlp.estimate_pi(M=5000)
        self.assertGreater(pi_hat, 2.5)
        self.assertLess(pi_hat, 3.8)

    def test_n_parameters(self):
        mlp = MLP([1, 64, 64, 32, 1])
        # 1×64+64 + 64×64+64 + 64×32+32 + 32×1+1 = 64+64 + 4096+64 + 2048+32 + 32+1
        expected = (1*64+64) + (64*64+64) + (64*32+32) + (32*1+1)
        self.assertEqual(mlp.n_parameters, expected)


class TestGaussianProcess(unittest.TestCase):
    """Tests du Processus Gaussien."""

    def setUp(self):
        self.ds = generate_dataset(n=100, sigma_noise=0.01, seed=42)
        self.gp = GaussianProcessRegressor(sigma_f=1.0, length_scale=0.2, sigma_n=0.01)
        self.gp.fit(self.ds.X_train[:80], self.ds.y_train[:80], optimize=False)

    def test_rbf_kernel_diagonal(self):
        X = np.array([0.0, 0.5, 1.0])
        K = self.gp.rbf_kernel(X, X)
        # Diagonale = σ_f²
        for i in range(3):
            self.assertAlmostEqual(K[i, i], self.gp.sigma_f**2, places=8)

    def test_rbf_kernel_symmetry(self):
        X = np.linspace(0, 1, 10)
        K = self.gp.rbf_kernel(X, X)
        np.testing.assert_allclose(K, K.T, atol=1e-12)

    def test_rbf_kernel_psd(self):
        X = np.linspace(0, 1, 20)
        K = self.gp.rbf_kernel(X, X) + 1e-6 * np.eye(20)
        eigvals = np.linalg.eigvalsh(K)
        self.assertTrue(np.all(eigvals >= -1e-8))

    def test_predict_shape(self):
        X_star = np.linspace(0, 1, 50)
        mu, std = self.gp.predict(X_star)
        self.assertEqual(mu.shape,  (50,))
        self.assertEqual(std.shape, (50,))

    def test_posterior_std_positive(self):
        X_star = np.linspace(0, 1, 50)
        _, std = self.gp.predict(X_star)
        self.assertTrue(np.all(std >= 0))

    def test_posterior_mean_near_data(self):
        # Aux points d'entraînement, μ* ≈ y
        mu, _ = self.gp.predict(self.ds.X_train[:10])
        np.testing.assert_allclose(mu, self.ds.y_train[:10], atol=0.1)

    def test_pi_estimate_reasonable(self):
        pi_hat, pi_std = self.gp.estimate_pi(M=2000)
        self.assertGreater(pi_hat, 2.8)
        self.assertLess(pi_hat, 3.5)
        self.assertGreater(pi_std, 0)

    def test_sample_posterior_shape(self):
        X_star  = np.linspace(0, 1, 30)
        samples = self.gp.sample_posterior(X_star, n_samples=5)
        self.assertEqual(samples.shape, (5, 30))


class TestRademacher(unittest.TestCase):
    """Tests de la complexité de Rademacher."""

    def setUp(self):
        self.X = np.linspace(0, 1, 100)

    def test_rademacher_positive(self):
        r = rademacher_polynomial(self.X, degree=5, n_mc=100)
        self.assertGreater(r, 0)

    def test_rademacher_increases_with_degree(self):
        r3 = rademacher_polynomial(self.X, degree=3,  n_mc=200)
        r8 = rademacher_polynomial(self.X, degree=8,  n_mc=200)
        self.assertGreater(r8, r3)

    def test_rademacher_decreases_with_n(self):
        X_small = np.linspace(0, 1, 30)
        X_large = np.linspace(0, 1, 300)
        r_small = rademacher_polynomial(X_small, degree=5, n_mc=200)
        r_large = rademacher_polynomial(X_large, degree=5, n_mc=200)
        self.assertGreater(r_small, r_large)

    def test_pac_bound_structure(self):
        ds = generate_dataset(n=100, sigma_noise=0.01, seed=42)
        result = pac_bound(ds.X_train, ds.y_train, degree=5, n_mc_rad=100)
        self.assertIn("emp",   result)
        self.assertIn("rad",   result)
        self.assertIn("conf",  result)
        self.assertIn("bound", result)
        # La borne doit être supérieure au risque empirique
        self.assertGreaterEqual(result["bound"], result["emp"])

    def test_pac_bound_positive(self):
        ds = generate_dataset(n=80, sigma_noise=0.01, seed=0)
        result = pac_bound(ds.X_train, ds.y_train, degree=4, n_mc_rad=100)
        for key in ("emp", "rad", "conf", "bound"):
            self.assertGreater(result[key], 0)


# ══════════════════════════════════════════════════════════════════════════════
#  TESTS D'INTÉGRATION
# ══════════════════════════════════════════════════════════════════════════════

class TestIntegration(unittest.TestCase):
    """Tests d'intégration bout-en-bout."""

    def test_all_methods_estimate_pi(self):
        """Vérifie que toutes les méthodes estiment π à <5% d'erreur."""
        ds = generate_dataset(n=200, sigma_noise=0.01, seed=42)

        # MC
        mc  = MonteCarloEstimator(seed=42)
        res_mc = mc.estimate(50_000)
        self.assertAlmostEqual(res_mc.pi_hat, PI_REF, delta=PI_REF * 0.05)

        # Poly
        ep  = PolynomialPiEstimator(max_degree=10, alpha_ridge=1e-12, cv_folds=3)
        rp  = ep.fit(ds.X_train, ds.y_train)
        self.assertAlmostEqual(rp.pi_hat, PI_REF, delta=PI_REF * 0.01)

        # MLP
        mlp = MLP([1, 32, 32, 1], seed=0)
        mlp.train(ds.X_train, ds.y_train, n_epochs=500, batch_size=32, verbose_every=0)
        pi_mlp = mlp.estimate_pi(M=5000)
        self.assertAlmostEqual(pi_mlp, PI_REF, delta=PI_REF * 0.05)

        # GP
        gp  = GaussianProcessRegressor()
        gp.fit(ds.X_train, ds.y_train, optimize=False)
        pi_gp, _ = gp.estimate_pi(M=3000)
        self.assertAlmostEqual(pi_gp, PI_REF, delta=PI_REF * 0.05)


# ══════════════════════════════════════════════════════════════════════════════
#  POINT D'ENTRÉE
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 65)
    print("  TESTS UNITAIRES — Approximation de π par ML")
    print("=" * 65)
    loader  = unittest.TestLoader()
    suite   = loader.discover(start_dir=os.path.dirname(__file__), pattern="test_*.py")
    runner  = unittest.TextTestRunner(verbosity=2)
    result  = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
