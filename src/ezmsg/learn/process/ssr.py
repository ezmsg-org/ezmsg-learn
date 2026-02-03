"""Self-supervised regression framework and LRR implementation.

This module provides a general framework for self-supervised channel
regression via :class:`SelfSupervisedRegressionTransformer`, and a
concrete implementation — Linear Regression Rereferencing (LRR) — via
:class:`LRRTransformer`.

**Framework.**  The base class accumulates the channel covariance
``C = X^T X`` and solves per-cluster ridge regressions to obtain a weight
matrix *W*.  Subclasses define what to *do* with *W* by implementing
:meth:`~SelfSupervisedRegressionTransformer._on_weights_updated` and
:meth:`~SelfSupervisedRegressionTransformer._process`.

**LRR.**  For each channel *c*, predict it from the other channels in its
cluster via ridge regression, then subtract the prediction::

    y = X - X @ W = X @ (I - W)

The effective weight matrix ``I - W`` is passed to
:class:`~ezmsg.sigproc.affinetransform.AffineTransformTransformer`, which
automatically exploits block-diagonal structure when ``channel_clusters``
are provided.

**Fitting.**  Given data matrix *X* of shape ``(samples, channels)``, the
sufficient statistic is the channel covariance ``C = X^T X``.  When
``incremental=True`` (default), *C* is accumulated across
:meth:`~SelfSupervisedRegressionTransformer.partial_fit` calls.

**Solving.**  Within each cluster the weight matrix *W* is obtained from
the inverse of the (ridge-regularised) cluster covariance
``C_inv = (C_cluster + lambda * I)^{-1}`` using the block-inverse identity::

    W[:, c] = -C_inv[:, c] / C_inv[c, c],    diag(W) = 0

This replaces the naive per-channel Cholesky loop with a single matrix
inverse per cluster, keeping the linear algebra in the source array
namespace so that GPU-backed arrays benefit from device-side computation.
"""

from __future__ import annotations

import os
import typing
from abc import abstractmethod
from pathlib import Path

import ezmsg.core as ez
import numpy as np
from array_api_compat import get_namespace
from ezmsg.baseproc import (
    BaseAdaptiveTransformer,
    BaseAdaptiveTransformerUnit,
    processor_state,
)
from ezmsg.baseproc.protocols import SettingsType, StateType
from ezmsg.sigproc.affinetransform import (
    AffineTransformSettings,
    AffineTransformTransformer,
)
from ezmsg.util.messages.axisarray import AxisArray

# ---------------------------------------------------------------------------
# Base: Self-supervised regression
# ---------------------------------------------------------------------------


class SelfSupervisedRegressionSettings(ez.Settings):
    """Settings common to all self-supervised regression modes."""

    weights: np.ndarray | str | Path | None = None
    """Pre-calculated weight matrix *W* or path to a CSV file (``np.loadtxt``
    compatible).  If provided, the transformer is ready immediately."""

    axis: str | None = None
    """Channel axis name. ``None`` defaults to the last dimension."""

    channel_clusters: list[list[int]] | None = None
    """Per-cluster regression.  ``None`` treats all channels as one cluster."""

    ridge_lambda: float = 0.0
    """Ridge (L2) regularisation parameter."""

    incremental: bool = True
    """When ``True``, accumulate ``X^T X`` across :meth:`partial_fit` calls.
    When ``False``, each call replaces the previous statistics."""


@processor_state
class SelfSupervisedRegressionState:
    cxx: object | None = None  # Array API; namespace matches source data.
    n_samples: int = 0
    weights: object | None = None  # Array API; namespace matches cxx.


class SelfSupervisedRegressionTransformer(
    BaseAdaptiveTransformer[SettingsType, AxisArray, AxisArray, StateType],
    typing.Generic[SettingsType, StateType],
):
    """Abstract base for self-supervised regression transformers.

    Subclasses must implement:

    * :meth:`_on_weights_updated` — called whenever the weight matrix *W* is
      (re)computed, so the subclass can build whatever internal transform it
      needs (e.g. ``I - W`` for LRR).
    * :meth:`_process` — the per-message transform step.
    """

    # -- message hash / state management ------------------------------------

    def _hash_message(self, message: AxisArray) -> int:
        axis = self.settings.axis or message.dims[-1]
        axis_idx = message.get_axis_idx(axis)
        return hash((message.key, message.data.shape[axis_idx]))

    def _reset_state(self, message: AxisArray) -> None:
        axis = self.settings.axis or message.dims[-1]
        axis_idx = message.get_axis_idx(axis)
        n_channels = message.data.shape[axis_idx]

        self._validate_clusters(n_channels)
        self._state.cxx = None
        self._state.n_samples = 0
        self._state.weights = None

        # If pre-calculated weights are provided, load and go.
        weights = self.settings.weights
        if weights is not None:
            if isinstance(weights, str):
                weights = Path(os.path.abspath(os.path.expanduser(weights)))
            if isinstance(weights, Path):
                weights = np.loadtxt(weights, delimiter=",")
            weights = np.asarray(weights, dtype=np.float64)
            self._state.weights = weights
            self._on_weights_updated()

    # -- cluster validation --------------------------------------------------

    def _validate_clusters(self, n_channels: int) -> None:
        """Raise if any cluster index is out of range."""
        clusters = self.settings.channel_clusters
        if clusters is None:
            return
        all_indices = np.concatenate([np.asarray(g) for g in clusters])
        if np.any((all_indices < 0) | (all_indices >= n_channels)):
            raise ValueError(f"channel_clusters contains out-of-range indices (valid range: 0..{n_channels - 1})")

    # -- weight solving ------------------------------------------------------

    def _solve_weights(self, cxx):
        """Solve all per-channel ridge regressions via matrix inverse.

        Uses the block-inverse identity: for target channel *c* with
        references *r*, ``w_c = -C_inv[r, c] / C_inv[c, c]`` where
        ``C_inv = (C_cluster + λI)⁻¹``.  This replaces the per-channel
        Cholesky loop with one matrix inverse per cluster.

        All computation stays in the source array namespace so that
        GPU-backed arrays benefit from device-side execution.  Cluster
        results are scattered into the full matrix via a selection-matrix
        multiply (``S @ W_cluster @ S^T``) to avoid numpy fancy indexing.

        Returns weight matrix *W* in the same namespace as *cxx*, with
        ``diag(W) == 0``.
        """
        xp = get_namespace(cxx)
        n = cxx.shape[0]

        clusters = self.settings.channel_clusters
        if clusters is None:
            clusters = [list(range(n))]

        W = xp.zeros((n, n), dtype=cxx.dtype)
        eye_n = xp.eye(n, dtype=cxx.dtype)

        for cluster in clusters:
            k = len(cluster)
            if k <= 1:
                continue

            idx_xp = xp.asarray(cluster)
            eye_k = xp.eye(k, dtype=cxx.dtype)

            # Extract cluster sub-covariance (stays on device)
            sub = xp.take(xp.take(cxx, idx_xp, axis=0), idx_xp, axis=1)

            if self.settings.ridge_lambda > 0:
                sub = sub + self.settings.ridge_lambda * eye_k

            # One inverse per cluster
            try:
                sub_inv = xp.linalg.inv(sub)
            except Exception:
                sub_inv = xp.linalg.pinv(sub)

            # Diagonal via element-wise product with identity
            diag_vals = xp.sum(sub_inv * eye_k, axis=0)

            # w_c = -C_inv[:, c] / C_inv[c, c], vectorised over all c
            W_cluster = -(sub_inv / xp.reshape(diag_vals, (1, k)))

            # Zero the diagonal
            W_cluster = W_cluster * (1.0 - eye_k)

            # Scatter into full W
            if k == n:
                W = W + W_cluster
            else:
                # Selection matrix: columns of eye(n) at cluster indices
                S = xp.take(eye_n, idx_xp, axis=1)  # (n, k)
                W = W + xp.matmul(S, xp.matmul(W_cluster, xp.permute_dims(S, (1, 0))))

        return W

    # -- partial_fit (self-supervised, accepts AxisArray) --------------------

    def partial_fit(self, message: AxisArray) -> None:  # type: ignore[override]
        xp = get_namespace(message.data)

        if xp.any(xp.isnan(message.data)):
            return

        # Hash check / state reset
        msg_hash = self._hash_message(message)
        if self._hash != msg_hash:
            self._reset_state(message)
            self._hash = msg_hash

        axis = self.settings.axis or message.dims[-1]
        axis_idx = message.get_axis_idx(axis)
        data = message.data

        # Move channel axis to last, flatten to 2-D
        if axis_idx != data.ndim - 1:
            perm = list(range(data.ndim))
            perm.append(perm.pop(axis_idx))
            data = xp.permute_dims(data, perm)

        n_channels = data.shape[-1]
        X = xp.reshape(data, (-1, n_channels))

        # Covariance stays in the source namespace for accumulation.
        cxx_new = xp.matmul(xp.permute_dims(X, (1, 0)), X)

        if self.settings.incremental and self._state.cxx is not None:
            self._state.cxx = self._state.cxx + cxx_new
        else:
            self._state.cxx = cxx_new
        self._state.n_samples += int(X.shape[0])

        self._state.weights = self._solve_weights(self._state.cxx)
        self._on_weights_updated()

    # -- convenience APIs ----------------------------------------------------

    def fit(self, X: np.ndarray) -> None:
        """Batch fit from a raw numpy array (samples x channels)."""
        n_channels = X.shape[-1]
        self._validate_clusters(n_channels)
        X = np.asarray(X, dtype=np.float64).reshape(-1, n_channels)
        self._state.cxx = X.T @ X
        self._state.n_samples = X.shape[0]
        self._state.weights = self._solve_weights(self._state.cxx)
        self._on_weights_updated()

    def fit_transform(self, message: AxisArray) -> AxisArray:
        """Convenience: ``partial_fit`` then ``_process``."""
        self.partial_fit(message)
        return self._process(message)

    # -- abstract hooks for subclasses ---------------------------------------

    @abstractmethod
    def _on_weights_updated(self) -> None:
        """Called after ``self._state.weights`` has been set/updated.

        Subclasses should build or refresh whatever internal transform
        object they need for :meth:`_process`.
        """
        ...

    @abstractmethod
    def _process(self, message: AxisArray) -> AxisArray: ...


# ---------------------------------------------------------------------------
# Concrete: Linear Regression Rereferencing (LRR)
# ---------------------------------------------------------------------------


class LRRSettings(SelfSupervisedRegressionSettings):
    """Settings for :class:`LRRTransformer`."""

    min_cluster_size: int = 32
    """Passed to :class:`AffineTransformTransformer` for the block-diagonal
    merge threshold."""


@processor_state
class LRRState(SelfSupervisedRegressionState):
    affine: AffineTransformTransformer | None = None


class LRRTransformer(
    SelfSupervisedRegressionTransformer[LRRSettings, LRRState],
):
    """Adaptive LRR transformer.

    ``partial_fit`` accepts a plain :class:`AxisArray` (self-supervised),
    and the transform step is delegated to an internal :class:`AffineTransformTransformer`.
    """

    # -- state management (clear own state, then delegate to base) ----------

    def _reset_state(self, message: AxisArray) -> None:
        self._state.affine = None
        super()._reset_state(message)

    # -- weights → affine transform -----------------------------------------

    def _on_weights_updated(self) -> None:
        xp = get_namespace(self._state.weights)
        n = self._state.weights.shape[0]
        effective = xp.eye(n, dtype=self._state.weights.dtype) - self._state.weights

        # Prefer in-place weight update when the affine transformer supports
        # it (avoids a full _reset_state round-trip on every partial_fit).
        if self._state.affine is not None:
            self._state.affine.set_weights(effective)
        else:
            self._state.affine = AffineTransformTransformer(
                AffineTransformSettings(
                    weights=effective,
                    axis=self.settings.axis,
                    channel_clusters=self.settings.channel_clusters,
                    min_cluster_size=self.settings.min_cluster_size,
                )
            )

    # -- transform -----------------------------------------------------------

    def _process(self, message: AxisArray) -> AxisArray:
        if self._state.affine is None:
            raise RuntimeError(
                "LRRTransformer has not been fitted. Call partial_fit() or provide pre-calculated weights."
            )
        return self._state.affine(message)


class LRRUnit(
    BaseAdaptiveTransformerUnit[
        LRRSettings,
        AxisArray,
        AxisArray,
        LRRTransformer,
    ],
):
    """ezmsg Unit wrapping :class:`LRRTransformer`.

    Follows the :class:`BaseAdaptiveDecompUnit` pattern — accepts
    :class:`AxisArray` (not :class:`SampleMessage`) for self-supervised
    training via ``INPUT_SAMPLE``.
    """

    SETTINGS = LRRSettings

    INPUT_SAMPLE = ez.InputStream(AxisArray)

    @ez.subscriber(INPUT_SAMPLE)
    async def on_sample(self, msg: AxisArray) -> None:
        await self.processor.apartial_fit(msg)
