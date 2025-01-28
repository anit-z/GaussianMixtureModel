import numpy as np
import librosa
import os
from tqdm import tqdm
from scipy.special import logsumexp

def custom_GMM_uni(data, K_components, epsilon, seed):
    np.random.seed(seed)#seed: int (for random initialization)
    n = len(data)# data: array_like (univariate, arbitrary dataset size)
    # For univariate data, we don't need to check dimensions as the operations
    # are designed for 1D arrays

    #  Initialize parameters with K_components
    omega = np.ones(K_components) / K_components # Weights array of size K
    mu = np.random.choice(data, K_components) # Means array of size K
    sigma = np.random.random(K_components) + 1e-2 # Variances array of size K

    prev_likelihood = -np.inf  # Initialize previous likelihood
    # E-step (Expectation) in univariate GMM:
    while True:
        # Calculate responsibilities (posterior probabilities)
        responsibilities = np.zeros((n, K_components))
        for k in range(K_components):
            responsibilities[:, k] = (
                omega[k] # Prior probability
                * (1 / np.sqrt(2 * np.pi * sigma[k])) # Normalization term
                * np.exp(-0.5 * ((data - mu[k]) ** 2) / sigma[k]) # Exponential term
            )

        # Normalize responsibilities
        responsibilities_sum = responsibilities.sum(axis=1, keepdims=True)
        responsibilities /= responsibilities_sum

        # M-step: Update parameters
        N_k = responsibilities.sum(axis=0) # Effective number of points per component
        omega = N_k / n # Update weights

        # Update means
        mu = np.zeros(K_components)
        for k in range(K_components):
            mu[k] = np.sum(responsibilities[:, k] * data) / N_k[k]

        # Update variances
        sigma = np.zeros(K_components)
        for k in range(K_components):
            sigma[k] = np.sum(responsibilities[:, k] * (data - mu[k])**2) / N_k[k]

        # Compute log-likelihood
        likelihood = np.sum(np.log(np.sum([
            omega[k] * (1 / np.sqrt(2 * np.pi * sigma[k])) *
            np.exp(-0.5 * ((data - mu[k]) ** 2) / sigma[k])
            for k in range(K_components)
        ], axis=0)))

        # Check convergence: epsilon: float (tolerance threshold)
        if np.abs(likelihood - prev_likelihood) < epsilon:
            break
        prev_likelihood = likelihood

    return {
        'omega': np.round(omega, 2),
        'mu': np.round(mu, 2),
        'sigma': np.round(sigma, 2)
    }#dictionary containing fitted model parameters, all rounded to 2 decimal places with np.round() {omega: fitted weights, mu: fitted means, Sigma: fitted variances}

def custom_GMM_multi(data, K_components, epsilon, seed):
    np.random.seed(seed)
    data = np.asarray(data)
    if data.ndim == 1:
        data = data.reshape(-1, 1)
    n_samples, n_features = data.shape

    # Initialize parameters
    omega = np.ones(K_components) / K_components
    random_indices = np.random.choice(n_samples, K_components, replace=False)
    mu = data[random_indices].copy()
    sigma = np.array([np.cov(data.T) + np.eye(n_features) * 1e-6 for _ in range(K_components)])

    prev_log_likelihood = -np.inf
    max_iter = 100

    for iteration in range(max_iter):
        # E-step: Calculate responsibilities
        responsibilities = np.zeros((n_samples, K_components))

        for k in range(K_components):
            diff = data - mu[k]
            try:
                inv_sigma = np.linalg.inv(sigma[k] + np.eye(n_features) * 1e-6)
            except np.linalg.LinAlgError:
                inv_sigma = np.linalg.pinv(sigma[k] + np.eye(n_features) * 1e-6)

            mahalanobis = np.sum(np.dot(diff, inv_sigma) * diff, axis=1)
            sign, logdet = np.linalg.slogdet(sigma[k] + np.eye(n_features) * 1e-6)
            log_prob = -0.5 * (n_features * np.log(2 * np.pi) + logdet + mahalanobis)
            responsibilities[:, k] = np.log(omega[k]) + log_prob

        # Log-sum-exp trick for numerical stability
        log_sum = logsumexp(responsibilities, axis=1)
        responsibilities = np.exp(responsibilities - log_sum[:, np.newaxis])

        # M-step: Update parameters
        Nk = np.sum(responsibilities, axis=0)
        omega = Nk / n_samples

        for k in range(K_components):
            if Nk[k] > 0:
                # Update means
                weighted_sum = np.sum(responsibilities[:, k:k+1] * data, axis=0)
                mu[k] = weighted_sum / Nk[k]

                # Update covariances
                diff = data - mu[k]
                weighted_diff = responsibilities[:, k:k+1] * diff
                sigma[k] = np.dot(weighted_diff.T, diff) / Nk[k]
                sigma[k] += np.eye(n_features) * 1e-6

        # Check convergence
        current_log_likelihood = np.sum(log_sum)
        if abs(current_log_likelihood - prev_log_likelihood) < epsilon:
            break
        prev_log_likelihood = current_log_likelihood

    return {'omega': omega, 'mu': mu, 'sigma': sigma}

def calculate_log_likelihood(data, gmm_params): #LLM suggested

    n_samples, n_features = data.shape
    log_likelihood = np.zeros(n_samples)

    for k in range(len(gmm_params['omega'])):
        diff = data - gmm_params['mu'][k]
        try:
            inv_sigma = np.linalg.inv(gmm_params['sigma'][k] + np.eye(n_features) * 1e-6)
        except np.linalg.LinAlgError:
            inv_sigma = np.linalg.pinv(gmm_params['sigma'][k] + np.eye(n_features) * 1e-6)

        mahalanobis = np.sum(np.dot(diff, inv_sigma) * diff, axis=1)
        sign, logdet = np.linalg.slogdet(gmm_params['sigma'][k] + np.eye(n_features) * 1e-6)
        log_prob = -0.5 * (n_features * np.log(2 * np.pi) + logdet + mahalanobis)
        log_likelihood = np.logaddexp(log_likelihood, np.log(gmm_params['omega'][k]) + log_prob)

    return np.sum(log_likelihood)

def speaker_rec_GMM(audio_dir, test_dir):

    def extract_mfcc(audio_path, n_mfcc=13):

        try:
            y, sr = librosa.load(audio_path, sr=16000)
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
            delta = librosa.feature.delta(mfcc)
            delta2 = librosa.feature.delta(mfcc, order=2)
            features = np.vstack([mfcc, delta, delta2])
            features = (features - np.mean(features, axis=1, keepdims=True)) / \
                      (np.std(features, axis=1, keepdims=True) + 1e-12)
            return features.T
        except Exception as e:
            print(f"Error extracting features from {audio_path}: {str(e)}")
            return None

    # Find speaker directories
    speaker_labels = []
    for root, _, _ in os.walk(audio_dir):
        current_dir = os.path.basename(root)
        if len(current_dir) == 3 and current_dir.isupper():
            speaker_labels.append(current_dir)

    if not speaker_labels:
        raise ValueError("No speaker directories found")

    # Train models
    speaker_models = {}
    pbar = tqdm(sorted(speaker_labels), desc="Training speaker models")

    for speaker in pbar:
        speaker_features = []

        # Collect features for current speaker
        for root, _, files in os.walk(audio_dir):
            if os.path.basename(root) == speaker:
                for file in files:
                    if file.endswith('.wav'):
                        mfcc = extract_mfcc(os.path.join(root, file))
                        if mfcc is not None and mfcc.shape[0] > 0:
                            speaker_features.append(mfcc)

        if speaker_features:
            try:
                X = np.vstack(speaker_features)
                if X.shape[0] > 0:
                    # Train GMM for current speaker
                    gmm_params = custom_GMM_multi(
                        data=X,
                        K_components=8,  # Number of Gaussian components
                        epsilon=1e-6,    # Convergence threshold
                        seed=42          # Random seed for reproducibility
                    )
                    speaker_models[speaker] = gmm_params
            except Exception as e:
                print(f"Warning: Could not train model for speaker {speaker}: {str(e)}")
                continue

    if not speaker_models:
        raise ValueError("No speaker models were successfully trained!")

    # Testing phase
    predict_dict = {}
    test_files = [f for f in os.listdir(test_dir) if f.endswith('.wav')]
    pbar = tqdm(test_files, desc="Testing files")

    default_speaker = list(speaker_models.keys())[0]

    for test_file in pbar:
        test_path = os.path.join(test_dir, test_file)
        try:
            # Extract features from test file
            test_mfcc = extract_mfcc(test_path)

            if test_mfcc is not None and test_mfcc.shape[0] > 0:
                # Calculate log likelihood for each speaker
                scores = {}
                for speaker, gmm_params in speaker_models.items():
                    scores[speaker] = calculate_log_likelihood(test_mfcc, gmm_params)

                if scores:
                    # Select speaker with highest log likelihood
                    predicted_speaker = max(scores.items(), key=lambda x: x[1])[0]
                    predict_dict[test_file] = predicted_speaker
                else:
                    predict_dict[test_file] = default_speaker
            else:
                predict_dict[test_file] = default_speaker

        except Exception as e:
            print(f"Warning: Could not process test file {test_file}: {str(e)}")
            predict_dict[test_file] = default_speaker

    return predict_dict