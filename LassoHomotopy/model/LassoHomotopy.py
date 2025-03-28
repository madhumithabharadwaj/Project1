import numpy as np

class LassoHomotopyModel():
    def __init__(self):
        self.coef_ = None

    def fit(self, X, y, max_iter=100, tol=1e-5):
        X = np.array(X, dtype=float)
        y = np.array(y, dtype=float).flatten()


        n_samples, n_features = X.shape
        beta = np.zeros(n_features)
        residual = y.copy()

        correlations = X.T @ residual
        abs_corr = np.abs(correlations)

        # Initialize: find first feature
        max_corr_idx = np.argmax(abs_corr)
        lambda_val = abs_corr[max_corr_idx]
        active_set = [max_corr_idx]

        print(f"Initial λ: {lambda_val}, Active feature: {max_corr_idx}")

        for iteration in range(max_iter):
            # Subset X for active features
            X_A = X[:, active_set]
            beta_A = np.linalg.pinv(X_A.T @ X_A) @ X_A.T @ y  # Use pinv for stability

            # Update full beta vector
            beta = np.zeros(n_features)
            beta[active_set] = beta_A

            # Update residual
            residual = y - X @ beta

            print(f"Iteration {iteration+1}, Beta: {beta}")

            # Compute new correlations
            correlations = X.T @ residual
            abs_corr = np.abs(correlations)

            # Mask out already active features
            for idx in active_set:
                abs_corr[idx] = 0

            max_corr_idx = np.argmax(abs_corr)
            max_corr = abs_corr[max_corr_idx]

            print(f"Max correlation: {max_corr} at feature {max_corr_idx}")

            if max_corr < tol:
                print("Converged.")
                break

            # Add new feature to active set
            active_set.append(max_corr_idx)

        self.coef_ = beta
        return LassoHomotopyResults(self.coef_)


class LassoHomotopyResults():
    def __init__(self, coef):
        self.coef_ = coef

    def predict(self, x):
        return np.dot(x, self.coef_)
