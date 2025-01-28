import numpy as np
from scipy.stats import multivariate_normal

def custom_GMM(data, K_components=2, epsilon=10e-6, seed=1234):
    """
    Implements Gaussian Mixture Model for multivariate data using EM algorithm

    Parameters:
    data: array-like, shape (N, D) where N is number of samples and D is dimensions
    K_components: number of Gaussian components (default=2)
    epsilon: convergence threshold (default=10e-6)
    seed: random seed for reproducibility

    Returns:
    dict containing:
        omega: array of component weights
        mu: array of component means
        Sigma: array of component covariance matrices
    """
    np.random.seed(seed)

    # Get dimensions
    N, D = data.shape

    # Initialize parameters
    omega = np.ones(K_components) / K_components

    # Initialize means using data statistics
    data_mean = np.mean(data, axis=0)
    data_cov = np.cov(data.T)
    mu = np.array([data_mean + np.random.multivariate_normal(np.zeros(D), data_cov)
                   for _ in range(K_components)])

    # Initialize covariance matrices
    Sigma = np.array([data_cov for _ in range(K_components)])

    prev_log_likelihood = float('-inf')

    while True:
        # E-step: Calculate responsibilities (gamma)
        gamma = np.zeros((N, K_components))

        for n in range(N):
            numerators = np.zeros(K_components)
            for k_ in range(K_components):
                # Using multivariate normal PDF
                numerators[k_] = omega[k_] * multivariate_normal.pdf(
                    data[n], mean=mu[k_], cov=Sigma[k_]
                )

            denominator = np.sum(numerators)
            gamma[n, :] = numerators / denominator

        # M-step:
        # Update weights (omega)
        N_k = np.sum(gamma, axis=0)
        omega = N_k / N

        # Update means (mu)
        for k_ in range(K_components):
            mu[k_] = np.sum(gamma[:, k_].reshape(-1, 1) * data, axis=0) / N_k[k_]

        # Update covariance matrices (Sigma)
        for k_ in range(K_components):
            diff = data - mu[k_]  # Shape: (N, D)
            # Compute outer products and weighted sum
            Sigma[k_] = np.zeros((D, D))
            for n in range(N):
                outer_prod = np.outer(diff[n], diff[n])  # Shape: (D, D)
                Sigma[k_] += gamma[n, k_] * outer_prod
            Sigma[k_] /= N_k[k_]

        # Calculate log likelihood
        current_log_likelihood = 0
        for n in range(N):
            sum_k = 0
            for k_ in range(K_components):
                sum_k += omega[k_] * multivariate_normal.pdf(
                    data[n], mean=mu[k_], cov=Sigma[k_]
                )
            current_log_likelihood += np.log(sum_k)

        # Check convergence
        if abs(current_log_likelihood - prev_log_likelihood) < epsilon:
            break

        prev_log_likelihood = current_log_likelihood

    return {
        'omega': omega,
        'mu': mu,
        'Sigma': Sigma
    }

if __name__ == "__main__":
    # Example usage with multivariate data
    # Generate sample data
    np.random.seed(42)
    N = 300  # number of samples
    D = 2    # dimensions

    # Generate synthetic multivariate data
    data = np.vstack([
        np.random.multivariate_normal([0, 0], [[1, 0.5], [0.5, 1]], N//2),
        np.random.multivariate_normal([4, 4], [[1.5, -0.2], [-0.2, 1.5]], N//2)
    ])

    # Fit GMM
    result = custom_GMM(data=data, K_components=2, epsilon=10e-6)
    print("\nFitted Parameters:")
    print("Weights (omega):", result['omega'])
    print("\nMeans (mu):")
    print(result['mu'])
    print("\nCovariance matrices (Sigma):")
    for k, cov in enumerate(result['Sigma']):
        print(f"\nComponent {k}:")
        print(cov)