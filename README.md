# GaussianMixtureModel
Implementation of Gaussian Mixture Models (Univariate and Multivariate) with examples in speech recognition and structured finance.

This repository contains an implementation of the **Gaussian Mixture Model (GMM)** for univariate and multivariate data using the Expectation-Maximization (EM) algorithm.

## Features
- Supports an arbitrary number of Gaussian components (`K_components`).
- Automatically initializes parameters.
- Supports convergence thresholds and reproducible results.

---

## **Mathematical Background**

### **Gaussian Probability Density Function (PDF)**

The Gaussian PDF is given by:

$$ f(x | \mu, \sigma^2) = \frac{1}{\sqrt{2\pi\sigma^2}} e^{-\frac{(x - \mu)^2}{2\sigma^2}} $$

Where:
- \( \mu \): Mean of the Gaussian distribution
- \( \sigma^2 \): Variance of the Gaussian distribution
- \( x \): Data point

---

### **The Expectation-Maximization (EM) Algorithm**

The **EM Algorithm** iteratively optimizes the parameters of the Gaussian Mixture Model through the following steps:

---

### **1. E-Step: Responsibilities (\( \gamma \))**

The responsibilities (\( \gamma_{nk} \)) for each component \( k \) and observation \( x_n \) are calculated as:

$$ \gamma_{nk} = \frac{\omega_k f(x_n | \mu_k, \sigma_k^2)}{\sum_{k'} \omega_{k'} f(x_n | \mu_{k'}, \sigma_{k'}^2)} $$

Where:
- \( \omega_k \): Weight of component \( k \)
- \( f(x_n | \mu_k, \sigma_k^2) \): Gaussian PDF for component \( k \)

---

### **2. M-Step: Parameter Updates**

Using the responsibilities from the E-step, the parameters are updated as follows:

#### **Update Component Weights (\( \omega_k \)):**

$$ \omega_k = \frac{\sum_n \gamma_{nk}}{N} $$

Where:
- \( N \): Total number of observations
- \( \gamma_{nk} \): Responsibility of component \( k \) for observation \( x_n \)

#### **Update Means (\( \mu_k \)):**

$$ \mu_k = \frac{\sum_n \gamma_{nk} x_n}{\sum_n \gamma_{nk}} $$

Where:
- \( x_n \): Observation \( n \)
- \( \gamma_{nk} \): Responsibility of component \( k \) for observation \( x_n \)

#### **Update Variances (\( \sigma_k^2 \)):**

$$ \sigma_k^2 = \frac{\sum_n \gamma_{nk} (x_n - \mu_k)^2}{\sum_n \gamma_{nk}} $$

Where:
- \( (x_n - \mu_k)^2 \): Squared deviation of observation \( n \) from the mean of component \( k \).

---

### **3. Log-Likelihood Calculation**

The log-likelihood is computed to assess convergence:

$$ \ln L(\omega, \mu, \sigma^2 | x) = \sum_n \ln \left( \sum_k \omega_k f(x_n | \mu_k, \sigma_k^2) \right) $$

Where:
- \( \ln L \): Log-likelihood of the data given the parameters
- \( f(x_n | \mu_k, \sigma_k^2) \): Gaussian PDF for component \( k \)
- \( \omega_k \): Weight of component \( k \)

---

### **Convergence Criterion**

The algorithm stops when the absolute change in log-likelihood between iterations is smaller than a predefined threshold (\( \epsilon \)):

$$ |\ln L_{\text{new}} - \ln L_{\text{old}}| < \epsilon $$

Where:
- \( \ln L_{\text{new}} \): The log-likelihood in the current iteration
- \( \ln L_{\text{old}} \): The log-likelihood from the previous iteration
- \( \epsilon \): Stopping threshold

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