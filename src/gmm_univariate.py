import numpy as np
from scipy.stats import norm

def custom_GMM_uni(data, K_components=2, epsilon=10e-6, seed=1234):
    """
    Implements Gaussian Mixture Model for univariate data using EM algorithm

    Parameters:
    data: array-like, univariate observations
    K_components: number of Gaussian components (default=2)
    epsilon: convergence threshold (default=10e-6)
    seed: random seed for reproducibility

    Returns:
    dict containing:
        omega: array of component weights
        mu: array of component means
        Sigma: array of component variances
    """
    # Set random seed
    np.random.seed(seed)

    # Initialize parameters
    N = len(data)

    # Initialize starting parameters
    # Set equal starting weights
    omega = np.ones(K_components) / K_components

    # Initialize means using data statistics
    data_mean = np.mean(data)
    data_var = np.var(data)
    mu = np.random.normal(data_mean, np.sqrt(data_var), K_components)

    # Set equal starting variance using total data variance
    Sigma = np.ones(K_components) * data_var

    # Initialize log likelihood
    prev_log_likelihood = float('-inf')

    while True:
        # E-step: Calculate responsibilities (gamma)
        gamma = np.zeros((N, K_components))

        for n in range(N):
            # Calculate numerators for all components
            numerators = np.zeros(K_components)
            for k_ in range(K_components):
                # Calculate PDF
                pdf = (1 / np.sqrt(2 * np.pi * Sigma[k_])) * \
                      np.exp(-(data[n] - mu[k_])**2 / (2 * Sigma[k_]))
                numerators[k_] = omega[k_] * pdf

            # Calculate responsibilities
            denominator = np.sum(numerators)
            for k_ in range(K_components):
                gamma[n, k_] = numerators[k_] / denominator

        # M-step:
        # Update weights (omega)
        N_k = np.sum(gamma, axis=0)
        for k_ in range(K_components):
            omega[k_] = N_k[k_] / N

        # Update means (mu)
        for k_ in range(K_components):
            numerator = 0
            for n in range(N):
                numerator += gamma[n, k_] * data[n]
            mu[k_] = numerator / N_k[k_]

        # Update variances (Sigma)
        for k_ in range(K_components):
            numerator = 0
            for n in range(N):
                diff = (data[n] - mu[k_])
                numerator += gamma[n, k_] * diff * diff
            Sigma[k_] = numerator / N_k[k_]

        # Calculate log likelihood
        current_log_likelihood = 0
        for n in range(N):
            sum_k = 0
            for k_ in range(K_components):
                pdf = (1 / np.sqrt(2 * np.pi * Sigma[k_])) * \
                      np.exp(-(data[n] - mu[k_])**2 / (2 * Sigma[k_]))
                sum_k += omega[k_] * pdf
            current_log_likelihood += np.log(sum_k)

        # Check convergence
        if abs(current_log_likelihood - prev_log_likelihood) < epsilon:
            break

        prev_log_likelihood = current_log_likelihood

    # Round the values to 2 decimal places
    omega = np.round(omega, 2)
    mu = np.round(mu, 2)
    Sigma = np.round(Sigma, 2)

    return {
        'omega': omega,
        'mu': mu,
        'Sigma': Sigma
    }

if __name__ == "__main__":
    # Example usage with any data
    result = custom_GMM_uni(data=your_data, K_components=2, epsilon=10e-6, seed=1234)
    print(result)