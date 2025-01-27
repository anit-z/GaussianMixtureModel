# GaussianMixtureModel
Implementation of Gaussian Mixture Models (Univariate and Multivariate) with examples in speech recognition and structured finance.

This repository contains an implementation of the **Gaussian Mixture Model (GMM)** for univariate and multivariate data using the Expectation-Maximization (EM) algorithm.

## Features
- Supports an arbitrary number of Gaussian components (`K_components`).
- Automatically initializes parameters.
- Supports convergence thresholds and reproducible results.

---

## Mathematical Background

### Gaussian Probability Density Function (PDF)

The Gaussian PDF is given by:

$$ f(x | \mu, \sigma^2) = \frac{1}{\sqrt{2\pi\sigma^2}} e^{-\frac{(x - \mu)^2}{2\sigma^2}} $$

Where:
- \( \mu \): Mean of the Gaussian distribution
- \( \sigma^2 \): Variance of the Gaussian distribution
- \( x \): Data point

### Expectation-Maximization (EM) Algorithm

The EM algorithm iteratively calculates:
1. **E-step**: Responsibilities (\( \gamma \)) for each component:
   $$ \gamma_{nk} = \frac{\omega_k f(x_n | \mu_k, \sigma_k^2)}{\sum_{k'} \omega_{k'} f(x_n | \mu_{k'}, \sigma_{k'}^2)} $$

2. **M-step**: Updates the parameters:
   - Component weights (\( \omega_k \)):
     $$ \omega_k = \frac{\sum_n \gamma_{nk}}{N} $$
   - Means (\( \mu_k \)):
     $$ \mu_k = \frac{\sum_n \gamma_{nk} x_n}{\sum_n \gamma_{nk}} $$
   - Variances (\( \sigma_k^2 \)):
     $$ \sigma_k^2 = \frac{\sum_n \gamma_{nk} (x_n - \mu_k)^2}{\sum_n \gamma_{nk}} $$

---

## Usage

### Code Example
```python
from custom_GMM import custom_GMM_uni

# Example data
data = [1.2, 2.3, 1.8, 3.4, 2.2]

# Run GMM
result = custom_GMM_uni(data, K_components=2, epsilon=1e-6, seed=1234)
print(result)