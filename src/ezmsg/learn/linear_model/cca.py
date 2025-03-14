import numpy as np
from scipy import linalg


class IncrementalCCA:
    def __init__(
        self,
        n_components=2,
        base_smoothing=0.95,
        min_smoothing=0.5,
        max_smoothing=0.99,
        adaptation_rate=0.1,
    ):
        """
        Parameters:
        -----------
        n_components : int
            Number of canonical components to compute
        base_smoothing : float
            Base smoothing factor (will be adapted)
        min_smoothing : float
            Minimum allowed smoothing factor
        max_smoothing : float
            Maximum allowed smoothing factor
        adaptation_rate : float
            How quickly to adjust smoothing factor (between 0 and 1)
        """
        self.n_components = n_components
        self.base_smoothing = base_smoothing
        self.current_smoothing = base_smoothing
        self.min_smoothing = min_smoothing
        self.max_smoothing = max_smoothing
        self.adaptation_rate = adaptation_rate
        self.initialized = False

    def initialize(self, d1, d2):
        """Initialize the necessary matrices"""
        self.d1 = d1
        self.d2 = d2

        # Initialize correlation matrices
        self.C11 = np.zeros((d1, d1))
        self.C22 = np.zeros((d2, d2))
        self.C12 = np.zeros((d1, d2))

        self.initialized = True

    def _compute_change_magnitude(self, C11_new, C22_new, C12_new):
        """Compute magnitude of change in correlation structure"""
        # Frobenius norm of differences
        diff11 = np.linalg.norm(C11_new - self.C11)
        diff22 = np.linalg.norm(C22_new - self.C22)
        diff12 = np.linalg.norm(C12_new - self.C12)

        # Normalize by matrix sizes
        diff11 /= self.d1 * self.d1
        diff22 /= self.d2 * self.d2
        diff12 /= self.d1 * self.d2

        return (diff11 + diff22 + diff12) / 3

    def _adapt_smoothing(self, change_magnitude):
        """Adapt smoothing factor based on detected changes"""
        # If change is large, decrease smoothing factor
        target_smoothing = self.base_smoothing * (1.0 - change_magnitude)
        target_smoothing = np.clip(
            target_smoothing, self.min_smoothing, self.max_smoothing
        )

        # Smooth the adaptation itself
        self.current_smoothing = (
            1 - self.adaptation_rate
        ) * self.current_smoothing + self.adaptation_rate * target_smoothing

    def partial_fit(self, X1, X2, update_projections=True):
        """Update the model with new samples using adaptive smoothing
        Assumes X1 and X2 are already centered and scaled"""
        if not self.initialized:
            self.initialize(X1.shape[1], X2.shape[1])

        # Compute new correlation matrices from current batch
        C11_new = X1.T @ X1 / X1.shape[0]
        C22_new = X2.T @ X2 / X2.shape[0]
        C12_new = X1.T @ X2 / X1.shape[0]

        # Detect changes and adapt smoothing factor
        if self.C11.any():  # Skip first update
            change_magnitude = self._compute_change_magnitude(C11_new, C22_new, C12_new)
            self._adapt_smoothing(change_magnitude)

        # Update with current smoothing factor
        alpha = self.current_smoothing
        self.C11 = alpha * self.C11 + (1 - alpha) * C11_new
        self.C22 = alpha * self.C22 + (1 - alpha) * C22_new
        self.C12 = alpha * self.C12 + (1 - alpha) * C12_new

        if update_projections:
            self._update_projections()

    def _update_projections(self):
        """Update canonical vectors and correlations"""
        eps = 1e-8
        C11_reg = self.C11 + eps * np.eye(self.d1)
        C22_reg = self.C22 + eps * np.eye(self.d2)

        K = (
            linalg.inv(linalg.sqrtm(C11_reg))
            @ self.C12
            @ linalg.inv(linalg.sqrtm(C22_reg))
        )
        U, self.correlations_, V = linalg.svd(K)

        self.x_weights_ = linalg.inv(linalg.sqrtm(C11_reg)) @ U[:, : self.n_components]
        self.y_weights_ = (
            linalg.inv(linalg.sqrtm(C22_reg)) @ V.T[:, : self.n_components]
        )

    def transform(self, X1, X2):
        """Project data onto canonical components"""
        X1_proj = X1 @ self.x_weights_
        X2_proj = X2 @ self.y_weights_
        return X1_proj, X2_proj
