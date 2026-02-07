"""Incremental Canonical Correlation Analysis (CCA).

.. note::
    This module supports the Array API standard via
    ``array_api_compat.get_namespace()``.  All linear algebra uses Array API
    operations; ``scipy.linalg.sqrtm`` is replaced by an eigendecomposition-
    based inverse square root (:func:`_inv_sqrtm_spd`).
"""

import numpy as np
from array_api_compat import get_namespace
from ezmsg.sigproc.util.array import array_device, xp_create


def _inv_sqrtm_spd(xp, A):
    """Inverse matrix square root for symmetric positive-definite matrices.

    Computes ``inv(sqrtm(A)) = Q @ diag(1/sqrt(lambda)) @ Q^T`` using the
    eigendecomposition.  This is more numerically stable than computing
    ``inv(sqrtm(...))`` separately and uses only Array API operations.
    """
    eigenvalues, eigenvectors = xp.linalg.eigh(A)
    eigenvalues = xp.clip(eigenvalues, 1e-12, None)  # avoid div-by-zero
    inv_sqrt_eig = 1.0 / xp.sqrt(eigenvalues)
    # Q @ diag(v) == Q * v (broadcasting), then @ Q^T
    return (eigenvectors * inv_sqrt_eig) @ xp.linalg.matrix_transpose(eigenvectors)


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

    def initialize(self, d1, d2, *, ref_array=None):
        """Initialize the necessary matrices.

        Args:
            d1: Dimensionality of the first dataset.
            d2: Dimensionality of the second dataset.
            ref_array: Optional reference array to derive array namespace
                and device from.  If ``None``, defaults to NumPy.
        """
        self.d1 = d1
        self.d2 = d2

        if ref_array is not None:
            xp = get_namespace(ref_array)
            dev = array_device(ref_array)
        else:
            xp, dev = np, None

        # Initialize correlation matrices
        self.C11 = xp_create(xp.zeros, (d1, d1), dtype=xp.float64, device=dev)
        self.C22 = xp_create(xp.zeros, (d2, d2), dtype=xp.float64, device=dev)
        self.C12 = xp_create(xp.zeros, (d1, d2), dtype=xp.float64, device=dev)

        self.initialized = True

    def _compute_change_magnitude(self, C11_new, C22_new, C12_new):
        """Compute magnitude of change in correlation structure."""
        xp = get_namespace(self.C11)

        # Frobenius norm of differences
        diff11 = xp.linalg.matrix_norm(C11_new - self.C11)
        diff22 = xp.linalg.matrix_norm(C22_new - self.C22)
        diff12 = xp.linalg.matrix_norm(C12_new - self.C12)

        # Normalize by matrix sizes
        diff11 = diff11 / (self.d1 * self.d1)
        diff22 = diff22 / (self.d2 * self.d2)
        diff12 = diff12 / (self.d1 * self.d2)

        return float((diff11 + diff22 + diff12) / 3)

    def _adapt_smoothing(self, change_magnitude):
        """Adapt smoothing factor based on detected changes."""
        # If change is large, decrease smoothing factor
        target_smoothing = self.base_smoothing * (1.0 - change_magnitude)
        target_smoothing = max(self.min_smoothing, min(target_smoothing, self.max_smoothing))

        # Smooth the adaptation itself
        self.current_smoothing = (
            1 - self.adaptation_rate
        ) * self.current_smoothing + self.adaptation_rate * target_smoothing

    def partial_fit(self, X1, X2, update_projections=True):
        """Update the model with new samples using adaptive smoothing.
        Assumes X1 and X2 are already centered and scaled."""
        xp = get_namespace(X1, X2)
        _mT = xp.linalg.matrix_transpose

        if not self.initialized:
            self.initialize(X1.shape[1], X2.shape[1], ref_array=X1)

        # Compute new correlation matrices from current batch
        C11_new = _mT(X1) @ X1 / X1.shape[0]
        C22_new = _mT(X2) @ X2 / X2.shape[0]
        C12_new = _mT(X1) @ X2 / X1.shape[0]

        # Detect changes and adapt smoothing factor
        if bool(xp.any(self.C11 != 0)):  # Skip first update
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
        """Update canonical vectors and correlations."""
        xp = get_namespace(self.C11)
        dev = array_device(self.C11)
        _mT = xp.linalg.matrix_transpose

        eps = 1e-8
        C11_reg = self.C11 + eps * xp_create(xp.eye, self.d1, dtype=self.C11.dtype, device=dev)
        C22_reg = self.C22 + eps * xp_create(xp.eye, self.d2, dtype=self.C22.dtype, device=dev)

        inv_sqrt_C11 = _inv_sqrtm_spd(xp, C11_reg)
        inv_sqrt_C22 = _inv_sqrtm_spd(xp, C22_reg)

        K = inv_sqrt_C11 @ self.C12 @ inv_sqrt_C22
        U, self.correlations_, Vh = xp.linalg.svd(K, full_matrices=False)

        self.x_weights_ = inv_sqrt_C11 @ U[:, : self.n_components]
        self.y_weights_ = inv_sqrt_C22 @ _mT(Vh)[:, : self.n_components]

    def transform(self, X1, X2):
        """Project data onto canonical components."""
        X1_proj = X1 @ self.x_weights_
        X2_proj = X2 @ self.y_weights_
        return X1_proj, X2_proj
