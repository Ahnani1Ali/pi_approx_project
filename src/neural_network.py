"""
src/neural_network.py
─────────────────────
Réseau de neurones MLP implémenté en NumPy pur (sans PyTorch/TensorFlow).

Architecture : Input(1) → [64] → [64] → [32] → Output(1)
Activation   : GELU (Gaussian Error Linear Unit, Hendrycks & Gimpel 2016)
Optimisation : Adam (Kingma & Ba 2014) avec mini-batches

Théorème fondamental :
    (Cybenko 1989, Hornik 1991) Pour σ non-polynomiale continue, ∀f∈C([0,1]),
    ∀ε>0, ∃ MLP à une couche cachée tel que ||f_θ - f||_∞ < ε.

    (Barron 1993) Pour f dans la classe de Barron (C_f < ∞), ∃ réseau à m
    neurones tel que MSE ≤ C_f²/m (sans malédiction de la dimensionnalité).

Références :
    - Cybenko (1989), Math Control Signals Syst
    - Kingma & Ba (2014), arXiv:1412.6980
    - Hendrycks & Gimpel (2016), arXiv:1606.08415
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Callable

PI_REF = np.pi


# ── Fonctions d'activation ─────────────────────────────────────────────────────

def gelu(x: np.ndarray) -> np.ndarray:
    """
    GELU : x * Φ(x)  avec approximation tanh (précision ~1e-4).
    Φ(x) = 0.5 * [1 + erf(x/√2)]
    """
    return x * 0.5 * (1.0 + np.tanh(
        np.sqrt(2.0 / np.pi) * (x + 0.044715 * x**3)
    ))

def gelu_grad(x: np.ndarray) -> np.ndarray:
    """Dérivée de GELU."""
    inner   = np.sqrt(2.0 / np.pi) * (x + 0.044715 * x**3)
    tanh_v  = np.tanh(inner)
    d_inner = np.sqrt(2.0 / np.pi) * (1.0 + 3.0 * 0.044715 * x**2)
    return 0.5 * (1.0 + tanh_v) + x * 0.5 * (1.0 - tanh_v**2) * d_inner

def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, x)

def relu_grad(x: np.ndarray) -> np.ndarray:
    return (x > 0).astype(float)

def tanh_act(x: np.ndarray) -> np.ndarray:
    return np.tanh(x)

def tanh_grad(x: np.ndarray) -> np.ndarray:
    return 1.0 - np.tanh(x)**2

ACTIVATIONS = {
    "gelu": (gelu, gelu_grad),
    "relu": (relu, relu_grad),
    "tanh": (tanh_act, tanh_grad),
}


@dataclass
class TrainingHistory:
    """Historique d'entraînement du réseau."""
    train_losses: list = field(default_factory=list)
    epochs:       list = field(default_factory=list)

    def add(self, epoch: int, loss: float):
        self.epochs.append(epoch)
        self.train_losses.append(loss)

    def final_loss(self) -> float:
        return self.train_losses[-1] if self.train_losses else float("inf")

    def convergence_rate(self) -> float:
        """Taux de réduction moyen de la perte (log-scale)."""
        if len(self.train_losses) < 2:
            return 0.0
        log_losses = np.log(np.maximum(self.train_losses, 1e-15))
        return (log_losses[-1] - log_losses[0]) / len(log_losses)


@dataclass
class MLPResult:
    """Résultat de l'estimation de π par MLP."""
    pi_hat:     float
    history:    TrainingHistory
    layer_sizes: list
    error_abs:  float = field(init=False)

    def __post_init__(self):
        self.error_abs = abs(self.pi_hat - PI_REF)

    def __str__(self) -> str:
        arch = " → ".join(str(s) for s in self.layer_sizes)
        return (
            f"MLP [{arch}]\n"
            f"  π̂        = {self.pi_hat:.10f}\n"
            f"  err       = {self.error_abs:.3e}\n"
            f"  loss fin. = {self.history.final_loss():.6e}\n"
            f"  n_epochs  = {len(self.history.epochs)}"
        )


class MLP:
    """
    Réseau de neurones multicouche (MLP) — NumPy pur.

    Paramètres
    ----------
    layer_sizes  : liste des dimensions [d_0, d_1, ..., d_L]
                   ex. [1, 64, 64, 32, 1]
    activation   : 'gelu' | 'relu' | 'tanh'
    seed         : graine pour l'initialisation

    Méthodes principales
    --------------------
    forward(X)        : propagation avant → (sortie, pre_activations, activations)
    backward(X, y)    : rétropropagation → (loss, gradients)
    adam_step(...)    : mise à jour Adam
    train(X, y, ...)  : boucle d'entraînement complète
    predict(X)        : prédiction sur nouvelles données
    estimate_pi(M)    : estimation de π par intégration numérique
    """

    def __init__(
        self,
        layer_sizes: list[int],
        activation: str = "gelu",
        seed: int = 0,
    ):
        self.layer_sizes = layer_sizes
        self.L = len(layer_sizes) - 1

        act_fn, act_grad = ACTIVATIONS.get(activation, ACTIVATIONS["gelu"])
        self._act      = act_fn
        self._act_grad = act_grad

        rng = np.random.default_rng(seed)

        # Initialisation He (Kaiming) : var(W) = 2/n_in
        self.W = []
        self.b = []
        for i in range(self.L):
            n_in, n_out = layer_sizes[i], layer_sizes[i + 1]
            std = np.sqrt(2.0 / n_in)
            self.W.append(rng.normal(0.0, std, (n_in, n_out)))
            self.b.append(np.zeros(n_out))

        # États du moment Adam (μ₁ et μ₂ pour W et b)
        self.mW = [np.zeros_like(w) for w in self.W]
        self.vW = [np.zeros_like(w) for w in self.W]
        self.mb = [np.zeros_like(b) for b in self.b]
        self.vb = [np.zeros_like(b) for b in self.b]
        self._t  = 0  # compteur de pas Adam

    @property
    def n_parameters(self) -> int:
        """Nombre total de paramètres entraînables."""
        return sum(w.size + b.size for w, b in zip(self.W, self.b))

    def forward(
        self, X: np.ndarray
    ) -> tuple[np.ndarray, list[np.ndarray], list[np.ndarray]]:
        """
        Propagation avant.

        Retourne : (output, pre_activations z, activations a)
        """
        a = X.reshape(-1, 1) if X.ndim == 1 else X
        zs, acts = [], []

        for i in range(self.L - 1):
            z = a @ self.W[i] + self.b[i]
            zs.append(z)
            a = self._act(z)
            acts.append(a)

        # Couche de sortie : activation linéaire (identité)
        z = a @ self.W[-1] + self.b[-1]
        zs.append(z)
        acts.append(z)

        return z, zs, acts

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Prédiction (mode inférence)."""
        out, _, _ = self.forward(X)
        return out.flatten()

    def mse_loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """Calcule la perte MSE sans rétropropagation."""
        y_hat = self.predict(X)
        return float(np.mean((y_hat - y)**2))

    def backward(
        self, X: np.ndarray, y: np.ndarray
    ) -> tuple[float, list[np.ndarray], list[np.ndarray]]:
        """
        Rétropropagation du gradient (perte MSE).

        Algorithme de rétropropagation standard :
          δ_L = 2(ŷ - y)/n                    [couche sortie]
          δ_l = (δ_{l+1} W_{l+1}ᵀ) ⊙ σ'(z_l) [couches cachées]
          ∂L/∂W_l = a_{l-1}ᵀ δ_l
          ∂L/∂b_l = Σᵢ δ_l[i]

        Retourne
        --------
        (loss, dW_list, db_list)
        """
        n     = len(y)
        X_in  = X.reshape(-1, 1) if X.ndim == 1 else X
        y_hat, zs, acts = self.forward(X_in)

        loss = float(np.mean((y_hat.flatten() - y)**2))

        # Gradient initial : ∂L/∂z_L = 2(ŷ-y)/n
        delta = 2.0 * (y_hat.flatten() - y).reshape(-1, 1) / n

        dW = [None] * self.L
        db = [None] * self.L

        # Couche de sortie
        a_prev    = acts[-2] if self.L > 1 else X_in
        dW[-1]    = a_prev.T @ delta
        db[-1]    = delta.sum(axis=0)

        # Couches cachées (de L-2 à 0, rétropropagation)
        for i in range(self.L - 2, -1, -1):
            delta   = (delta @ self.W[i + 1].T) * self._act_grad(zs[i])
            a_prev  = acts[i - 1] if i > 0 else X_in
            dW[i]   = a_prev.T @ delta
            db[i]   = delta.sum(axis=0)

        return loss, dW, db

    def adam_step(
        self,
        dW: list[np.ndarray],
        db: list[np.ndarray],
        lr: float     = 5e-4,
        beta1: float  = 0.9,
        beta2: float  = 0.999,
        eps: float    = 1e-8,
    ) -> None:
        """
        Mise à jour Adam (Kingma & Ba 2014).

        m_t = β₁ m_{t-1} + (1-β₁) g_t
        v_t = β₂ v_{t-1} + (1-β₂) g_t²
        m̂_t = m_t / (1-β₁ᵗ)      [correction du biais]
        v̂_t = v_t / (1-β₂ᵗ)
        θ_t = θ_{t-1} - η m̂_t / (√v̂_t + ε)
        """
        self._t += 1
        bc1 = 1.0 - beta1**self._t
        bc2 = 1.0 - beta2**self._t

        for i in range(self.L):
            # Poids W
            self.mW[i] = beta1 * self.mW[i] + (1.0 - beta1) * dW[i]
            self.vW[i] = beta2 * self.vW[i] + (1.0 - beta2) * dW[i]**2
            self.W[i] -= lr * (self.mW[i] / bc1) / (np.sqrt(self.vW[i] / bc2) + eps)

            # Biais b
            self.mb[i] = beta1 * self.mb[i] + (1.0 - beta1) * db[i]
            self.vb[i] = beta2 * self.vb[i] + (1.0 - beta2) * db[i]**2
            self.b[i] -= lr * (self.mb[i] / bc1) / (np.sqrt(self.vb[i] / bc2) + eps)

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_epochs: int      = 4000,
        batch_size: int    = 64,
        lr: float          = 5e-4,
        beta1: float       = 0.9,
        beta2: float       = 0.999,
        eps: float         = 1e-8,
        verbose_every: int = 1000,
        early_stop_tol: float = 1e-9,
        early_stop_patience: int = 200,
    ) -> TrainingHistory:
        """
        Boucle d'entraînement : SGD-Adam avec mini-batches.

        Paramètres
        ----------
        X, y             : données d'entraînement
        n_epochs         : nombre d'époques
        batch_size       : taille des mini-lots
        lr               : taux d'apprentissage initial
        verbose_every    : fréquence d'affichage
        early_stop_tol   : tolérance arrêt anticipé sur la perte
        early_stop_patience : n. d'époques sans amélioration avant arrêt

        Retourne
        --------
        TrainingHistory
        """
        n       = len(X)
        history = TrainingHistory()
        rng     = np.random.default_rng(99)
        best_loss = float("inf")
        patience_counter = 0

        for epoch in range(n_epochs):
            # Mélange des données à chaque époque
            idx    = rng.permutation(n)
            X_shuf = X[idx]
            y_shuf = y[idx]

            epoch_loss = 0.0
            n_batches  = 0

            for start in range(0, n, batch_size):
                Xb = X_shuf[start:start + batch_size]
                yb = y_shuf[start:start + batch_size]
                loss, dW, db = self.backward(Xb, yb)
                self.adam_step(dW, db, lr=lr, beta1=beta1, beta2=beta2, eps=eps)
                epoch_loss += loss
                n_batches  += 1

            avg_loss = epoch_loss / max(n_batches, 1)
            history.add(epoch + 1, avg_loss)

            if verbose_every and (epoch + 1) % verbose_every == 0:
                print(f"  Époque {epoch+1:5d}/{n_epochs}  —  loss MSE : {avg_loss:.6e}")

            # Arrêt anticipé
            if avg_loss < best_loss - early_stop_tol:
                best_loss = avg_loss
                patience_counter = 0
            else:
                patience_counter += 1
            if patience_counter >= early_stop_patience:
                if verbose_every:
                    print(f"  Arrêt anticipé à l'époque {epoch+1} (loss={avg_loss:.2e})")
                break

        return history

    def estimate_pi(self, M: int = 50_000, method: str = "trapz") -> float:
        """
        Estime π par intégration numérique de f̂_θ.

        π̂ = 4 ∫₀¹ f̂_θ(x) dx

        Paramètres
        ----------
        M      : nombre de points d'intégration
        method : 'trapz' (trapèzes) ou 'simpson' (Simpson)

        Retourne
        --------
        float : estimation de π
        """
        x_quad = np.linspace(0.0, 1.0, M)
        f_pred = self.predict(x_quad)

        if method == "trapz":
            return 4.0 * np.trapezoid(f_pred, x_quad)
        elif method == "simpson":
            from scipy.integrate import simpson
            return 4.0 * simpson(f_pred, x=x_quad)
        else:
            return 4.0 * np.trapezoid(f_pred, x_quad)

    def fit_and_estimate(
        self, X: np.ndarray, y: np.ndarray, **train_kwargs
    ) -> MLPResult:
        """Raccourci : entraîne et retourne un MLPResult."""
        history = self.train(X, y, **train_kwargs)
        pi_hat  = self.estimate_pi()
        return MLPResult(
            pi_hat=pi_hat,
            history=history,
            layer_sizes=self.layer_sizes,
        )


if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from src.dataset import generate_dataset

    ds  = generate_dataset(n=300, sigma_noise=0.01, seed=42)

    print("Entraînement du MLP...")
    mlp = MLP(layer_sizes=[1, 64, 64, 32, 1], activation="gelu", seed=7)
    res = mlp.fit_and_estimate(
        ds.X_train, ds.y_train,
        n_epochs=4000, batch_size=64, lr=5e-4, verbose_every=1000
    )
    print(res)
    print(f"\nNombre de paramètres : {mlp.n_parameters:,}")

    # Comparaison des méthodes d'intégration
    print(f"\nTrapèzes  : π̂ = {mlp.estimate_pi(M=50000, method='trapz'):.10f}")
    print(f"Simpson   : π̂ = {mlp.estimate_pi(M=50000, method='simpson'):.10f}")
