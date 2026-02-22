# Approximation de π par Machine Learning

---

## Structure du Projet

```
pi_approx_project/
├── main.py                    # Script principal (toutes les expériences)
├── requirements.txt           # Dépendances Python
├── README.md
│
├── configs/
│   ├── __init__.py
│   └── config.py              # Paramètres centralisés (seeds, hyperparamètres)
│
├── src/
│   ├── __init__.py
│   ├── dataset.py             # Génération du dataset synthétique
│   ├── monte_carlo.py         # Estimateur Monte-Carlo + TCL + variable de contrôle
│   ├── polynomial_regression.py  # Régression polynomiale Ridge + CV + Gauss-Legendre
│   ├── neural_network.py      # MLP NumPy pur (Adam, GELU, rétropropagation)
│   ├── gaussian_process.py    # GP + noyau RBF + MLE-II + postérieure exacte
│   ├── rademacher.py          # Complexité de Rademacher + bornes PAC
│   └── visualization.py      # Figures publication-quality
│
├── tests/
│   ├── __init__.py
│   └── test_all.py            # 35+ tests unitaires + tests d'intégration
│
├── results/
│   ├── figures/               # Figures PNG (générées par main.py)
│   └── data/                  # CSV des résultats numériques
│
└── docs/
    └── rapport_approximation_pi_ML.pdf   # Rapport LaTeX (19 pages)
```

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Utilisation

### Exécuter toutes les expériences
```bash
python main.py
```

### Mode rapide (tests, moins d'époques)
```bash
python main.py --fast
```

### Une seule méthode
```bash
python main.py --method mc    # Monte-Carlo
python main.py --method poly  # Régression polynomiale
python main.py --method mlp   # Réseau de neurones
python main.py --method gp    # Processus Gaussien
```

### Tests unitaires
```bash
python -m pytest tests/ -v
# ou directement :
python tests/test_all.py
```

---

## Méthodes Implémentées

| Méthode | Fichier | Taux de convergence | Hypothèses |
|---|---|---|---|
| Monte-Carlo | `src/monte_carlo.py` | $O(n^{-1/2})$ | Aucune |
| Régression polynomiale | `src/polynomial_regression.py` | $O(\rho^{-d})$ | $f$ analytique |
| MLP (NumPy pur) | `src/neural_network.py` | $O(m^{-1})$ (Barron) | $f \in$ classe de Barron |
| Processus Gaussien | `src/gaussian_process.py` | Expo. en régularité | $f \in$ RKHS |

---

## Fondements Mathématiques

### Principe géométrique

$$
\pi = 4 \int_{0}^{1} \sqrt{1 - x^{2}} \, dx
$$

Le dataset est généré depuis  
$y_i = \sqrt{1 - x_i^{2}} + \varepsilon_i$,  
$\varepsilon_i \sim \mathcal{N}(0, \sigma^{2})$.

### Monte-Carlo
$$\hat{\pi}_n^{MC} = \frac{4}{n}\sum_{i=1}^n \mathbf{1}_{U_{1,i}^2+U_{2,i}^2\leq 1}, \quad \sqrt{n}(\hat{\pi}_n - \pi) \xrightarrow{\mathcal{D}} \mathcal{N}(0, \pi(4-\pi))$$

### Régression Ridge
$$\hat{\theta} = (\Phi^\top\Phi + \lambda I)^{-1}\Phi^\top y, \quad \hat{\pi} = 4\sum_{k=0}^d \frac{\hat{\theta}_k}{k+1}$$

### MLP (GELU + Adam)
$$\text{GELU}(z) = z \cdot \Phi(z), \quad f_\theta(x) \approx \sqrt{1-x^2}, \quad \hat{\pi} = 4\int_0^1 f_\theta(x)\,dx$$

### Processus Gaussien
$$
\mu(x)=\mathbf{k}(x)^\top(\mathbf{K}+\sigma_n^2 \mathbf{I})^{-1}\mathbf{y},
\qquad
\hat{\pi}=4\int_0^1 \mu(x)\,dx
$$

### Borne PAC (Bartlett & Mendelson 2002)
$$\mathcal{R}(f) \leq \hat{\mathcal{R}}_n(f) + 2\hat{\mathfrak{R}}_n(\mathcal{H}) + \sqrt{\frac{\log(1/\delta)}{2n}}$$

---

## Résultats Produits

Après `python main.py`, les fichiers suivants sont générés :

### Figures (`results/figures/`)
- `fig_dataset.png`     — Dataset et interprétation géométrique
- `fig_montecarlo.png`  — Points MC, convergence, validation TCL
- `fig_polynomial.png`  — Ajustement, sélection de degré, conditionnement
- `fig_mlp.png`         — Courbe de perte, ajustement, résidus
- `fig_gp.png`          — Postérieure GP, incertitude, tirages
- `fig_comparison.png`  — Comparaison globale des méthodes
- `fig_rademacher.png`  — Complexité de Rademacher et bornes PAC

### Données (`results/data/`)
- `monte_carlo_convergence.csv`  — Erreur MC vs n
- `polynomial_cv_scores.csv`     — MSE par degré (validation croisée)
- `mlp_training_loss.csv`        — Courbe de perte par époque
- `gp_posterior.csv`             — Moyenne/variance postérieure GP
- `convergence_comparison.csv`   — Convergence de toutes les méthodes
- `final_results.csv`            — Tableau récapitulatif final

---

## Références

1. Cybenko, G. (1989). Approximation by superpositions of a sigmoidal function. *Math. Control Signals Syst.*
2. Hornik, K. (1991). Approximation capabilities of multilayer feedforward networks. *Neural Networks.*
3. Barron, A. R. (1993). Universal approximation bounds. *IEEE Trans. Inf. Theory.*
4. Kingma, D. P., & Ba, J. (2014). Adam: A method for stochastic optimization. *ICLR 2015.*
5. Rasmussen, C. E., & Williams, C. K. I. (2006). *Gaussian Processes for Machine Learning.* MIT Press.
6. Bartlett, P. L., & Mendelson, S. (2002). Rademacher and Gaussian complexities. *JMLR.*
7. Hendrycks, D., & Gimpel, K. (2016). Gaussian Error Linear Units (GELUs). *arXiv:1606.08415.*
8. Trefethen, L. N. (2019). *Approximation Theory and Approximation Practice.* SIAM.

