"""Self-supervised regression framework and LRR implementation.

This module provides a general framework for self-supervised channel
regression via :class:`SelfSupervisedRegressionTransformer`, and a
concrete implementation — Linear Regression Rereferencing (LRR) — via
:class:`LRRTransformer`.

**Framework.**  The base class accumulates the channel covariance
``C = X^T X`` and solves per-group ridge regressions to obtain a weight
matrix *W*.  Subclasses define what to *do* with *W* by implementing
:meth:`~SelfSupervisedRegressionTransformer._on_weights_updated` and
:meth:`~SelfSupervisedRegressionTransformer._process`.

**LRR.**  For each channel *c*, predict it from the other channels in its
group via ridge regression, then subtract the prediction::

    y = X - X @ W = X @ (I - W)

The effective weight matrix ``I - W`` is passed to
:class:`~ezmsg.sigproc.affinetransform.AffineTransformTransformer`, which
reads the block-diagonal structure off the weight matrix itself and picks
a dense or block matmul accordingly — the channel grouping is an input to
*fitting* only, never to applying.

**Fitting.**  Given data matrix *X* of shape ``(samples, channels)``, the
sufficient statistic is the channel covariance ``C = X^T X``.  When
``incremental=True`` (default), *C* is accumulated across
:meth:`~SelfSupervisedRegressionTransformer.partial_fit` calls.

**Solving.**  Within each group the weight matrix *W* is obtained from
the inverse of the (ridge-regularised) group covariance
``C_inv = (C_group + lambda * I)^{-1}`` using the block-inverse identity::

    W[:, c] = -C_inv[:, c] / C_inv[c, c],    diag(W) = 0

This replaces the naive per-channel Cholesky loop with a single matrix
inverse per group, keeping the linear algebra in the source array
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
from ezmsg.sigproc.util.array import array_device, xp_create
from ezmsg.sigproc.util.channels import (
    ChannelGroupSpec,
    group_spec_fields,
    group_spec_fingerprint,
    resolve_channel_groups,
    validate_channel_groups,
)
from ezmsg.sigproc.util.rereference import RereferenceKind, rereference_matrix
from ezmsg.util.messages.axisarray import AxisArray

# Minimum channels a group needs before it is rereferenced. Rereferencing
# regresses each channel against the *others* in its group, so a group with
# fewer than this many channels has too few references to be meaningful (1 -> no
# reference at all; 2 -> a single, degenerate reference). Such groups are passed
# through untouched (identity). This also makes sliced/partial inputs robust: a
# group reduced to a channel or two (or an empty group) is a no-op rather than
# a crash or an unstable fit. Kept a module const for now; promote to a setting if
# callers need to tune it.
MIN_REREF_GROUP_SIZE = 3


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

    channel_groups: ChannelGroupSpec | None = None
    """How to split the channel axis into groups for per-group regression: explicit
    index groups (``[[0, 1, 2], [3, 4, 5]]``), the name of a structured field on the
    channel coordinate axis (``"bank"`` to regress within each electrode bank), a
    tuple of field names, or a callable. See
    :data:`~ezmsg.sigproc.util.channels.ChannelGroupSpec`.

    ``None`` -- or a field spec the incoming axis doesn't carry -- falls back to
    ``block_size``, then to a single all-channel group."""

    block_size: int | None = None
    """Fallback grouping when ``channel_groups`` is ``None`` or resolves to nothing:
    consecutive blocks of this many channels."""

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
    resolved_groups: list | None = None
    """``channel_groups`` resolved against the message at reset, cached so
    ``_get_channel_groups`` can return them without one."""


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
        # group_spec_fingerprint contributes an O(1) "can this spec resolve?"
        # boolean rather than the field's bytes, so the per-message hash does not
        # grow with channel count. See its docstring for what that deliberately
        # does not detect. Mirrors the ezmsg-sigproc transformers' hash.
        return hash(
            (message.key, message.data.shape[axis_idx])
            + group_spec_fingerprint(message, axis, self.settings.channel_groups)
        )

    def _reset_state(self, message: AxisArray) -> None:
        axis = self.settings.axis or message.dims[-1]
        axis_idx = message.get_axis_idx(axis)
        n_channels = message.data.shape[axis_idx]

        # Resolve the grouping against this message (a field- or callable-based
        # spec needs one). Cached so the message-less _get_channel_groups can
        # return it later.
        self._state.resolved_groups = resolve_channel_groups(message, axis, self.settings.channel_groups)

        self._validate_groups(n_channels)
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

    # -- group resolution / validation ---------------------------------------

    def _static_channel_groups(self) -> list[np.ndarray] | None:
        """Groups readable from the settings alone, without a message.

        Explicit index groups are static; field-name and callable specs depend on
        the message and are resolved in :meth:`_reset_state` instead. This is what
        lets the message-less :meth:`fit` honour explicit groups.
        """
        spec = self.settings.channel_groups
        if spec is None or callable(spec) or group_spec_fields(spec) is not None:
            return None
        return [np.asarray(group, dtype=np.intp).reshape(-1) for group in spec]

    def _get_channel_groups(self, n_channels: int) -> list[np.ndarray] | None:
        # Precedence: resolved channel_groups (cached at reset, or static explicit
        # groups) > block_size > None (single all-channel group).
        groups = self._state.resolved_groups
        if groups is None:
            groups = self._static_channel_groups()
        if groups is None and self.settings.block_size is not None:
            groups = [
                np.arange(i, min(i + self.settings.block_size, n_channels), dtype=np.intp)
                for i in range(0, n_channels, self.settings.block_size)
            ]
        return groups

    def _validate_groups(self, n_channels: int) -> None:
        """Raise if the resolved groups are empty, out of range, or overlapping."""
        groups = self._get_channel_groups(n_channels)
        if groups is None:
            return  # implicit single group
        if len(groups) == 0:
            # An empty group list is only legitimate with no channels (e.g. a
            # fully sliced-out input). With channels present it means an explicit
            # channel_groups=[], which would silently disable rereferencing --
            # fail fast instead.
            if n_channels == 0:
                return
            raise ValueError(
                f"channel_groups is empty but the input has {n_channels} channels. "
                "Pass channel_groups=None to treat all channels as a single "
                "group, or provide non-empty channel index groups."
            )
        validate_channel_groups(groups, n_channels)

    # -- weight solving ------------------------------------------------------

    def _solve_weights(self, cxx):
        """Solve all per-channel ridge regressions via matrix inverse.

        Uses the block-inverse identity: for target channel *c* with
        references *r*, ``w_c = -C_inv[r, c] / C_inv[c, c]`` where
        ``C_inv = (C_group + λI)⁻¹``.  This replaces the per-channel
        Cholesky loop with one matrix inverse per group.

        All computation stays in the source array namespace so that
        GPU-backed arrays benefit from device-side execution.  Group
        results are scattered into the full matrix via a selection-matrix
        multiply (``S @ W_group @ S^T``) to avoid numpy fancy indexing.

        Returns weight matrix *W* in the same namespace as *cxx*, with
        ``diag(W) == 0``.
        """
        xp = get_namespace(cxx)
        dev = array_device(cxx)
        n = cxx.shape[0]

        groups = self._get_channel_groups(n)
        if groups is None:
            groups = [np.arange(n, dtype=np.intp)]

        W = xp_create(xp.zeros, (n, n), dtype=cxx.dtype, device=dev)
        eye_n = xp_create(xp.eye, n, dtype=cxx.dtype, device=dev)

        # MLX linalg ops are CPU-only; with unified memory the explicit CPU
        # stream is a scheduling hint, not a host copy, and results stay mlx.
        inv_kwargs = {"stream": xp.cpu} if xp.__name__ == "mlx.core" else {}

        for group in groups:
            idx = np.asarray(group, dtype=np.intp).reshape(-1)
            k = idx.size
            if k < MIN_REREF_GROUP_SIZE:
                # Too few channels to rereference against -- leave these channels
                # untouched (W rows stay 0 -> identity). Covers sliced/partial
                # groups down to a single channel; never raises.
                continue

            idx_list = idx.tolist()
            idx_xp = xp.asarray(idx_list) if dev is None else xp.asarray(idx_list, device=dev)
            eye_k = xp_create(xp.eye, k, dtype=cxx.dtype, device=dev)

            # Extract group sub-covariance (stays on device)
            sub = xp.take(xp.take(cxx, idx_xp, axis=0), idx_xp, axis=1)

            if self.settings.ridge_lambda > 0:
                sub = sub + self.settings.ridge_lambda * eye_k

            # One inverse per group
            try:
                sub_inv = xp.linalg.inv(sub, **inv_kwargs)
            except Exception:
                sub_inv = xp.linalg.pinv(sub, **inv_kwargs)

            # Diagonal via element-wise product with identity
            diag_vals = xp.sum(sub_inv * eye_k, axis=0)

            # w_c = -C_inv[:, c] / C_inv[c, c], vectorised over all c
            W_group = -(sub_inv / xp.reshape(diag_vals, (1, k)))

            # Zero the diagonal
            W_group = W_group * (1.0 - eye_k)

            # Scatter into full W. The no-op shortcut needs the group to be every
            # channel *in order* -- a callable spec may return all n permuted, and
            # then the sub-block still has to be scattered back.
            if k == n and np.array_equal(idx, np.arange(n, dtype=np.intp)):
                W = W + W_group
            else:
                # Selection matrix: columns of eye(n) at group indices
                S = xp.take(eye_n, idx_xp, axis=1)  # (n, k)
                W = W + xp.matmul(S, xp.matmul(W_group, xp.permute_dims(S, (1, 0))))

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
        if n_channels == 0:
            # No channels to fit (e.g. a fully sliced-out hub). Leave the weights
            # untouched; _process passes the 0-channel data through unchanged.
            return
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
        self._validate_groups(n_channels)
        if n_channels == 0:
            # No channels to fit -- same 0-channel no-op as partial_fit.
            return
        X = np.asarray(X, dtype=np.float64).reshape(-1, n_channels)
        self._state.cxx = X.T @ X
        self._state.n_samples = X.shape[0]
        self._state.weights = self._solve_weights(self._state.cxx)
        self._on_weights_updated()

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

    kernel: str = "auto"
    """Forwarded to :attr:`~ezmsg.sigproc.affinetransform.AffineTransformSettings.kernel`.
    ``"auto"`` lets the affine transformer choose between a dense and a
    block-diagonal matmul from the structure of ``I - W``; ``"dense"`` /
    ``"blocks"`` force it."""

    init_default: RereferenceKind = RereferenceKind.IDENTITY
    """Effective transform used when ``weights`` is None and nothing has been fit
    yet. ``IDENTITY`` passes through (legacy); ``CAR`` applies per-group
    leave-one-out common-average referencing from the resolved groups (groups
    below :data:`MIN_REREF_GROUP_SIZE` stay identity, matching the fit's
    passthrough). Provided or fitted weights always take precedence over this
    cold-start default."""


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
        dev = array_device(self._state.weights)
        n = self._state.weights.shape[0]
        effective = xp_create(xp.eye, n, dtype=self._state.weights.dtype, device=dev) - self._state.weights

        # Prefer in-place weight update when the affine transformer supports it
        # (avoids a full _reset_state round-trip on every partial_fit). The
        # default recalc_structure=False is what we want: refitting changes the
        # weight *values*, not their sparsity pattern, which is fixed by the
        # channel grouping.
        if self._state.affine is not None:
            self._state.affine.set_weights(effective)
        else:
            # No channel_groups: the affine derives block structure from the
            # weight matrix itself, and grouping only ever builds kind/callable
            # weights -- which these are not.
            self._state.affine = AffineTransformTransformer(
                AffineTransformSettings(
                    weights=effective,
                    axis=self.settings.axis,
                    kernel=self.settings.kernel,
                )
            )

    # -- transform -----------------------------------------------------------

    def _process(self, message: AxisArray) -> AxisArray:
        axis = self.settings.axis or message.dims[-1]
        if message.data.shape[message.get_axis_idx(axis)] == 0:
            # No channels (e.g. a fully sliced-out hub): nothing to rereference.
            # Pass the 0-channel message through unchanged -- building an affine
            # from empty channel groups would raise downstream.
            return message
        if self._state.affine is None:
            axis_idx = message.get_axis_idx(axis)
            n_channels = message.data.shape[axis_idx]

            # No weights provided or fit yet: build the configured cold-start
            # default (identity, or per-group leave-one-out CAR matching the
            # fit's passthrough for groups below MIN_REREF_GROUP_SIZE).
            # Built as numpy; the affine transformer converts weights to the
            # message's namespace/dtype/device on first use.
            groups = self._get_channel_groups(n_channels)
            effective = rereference_matrix(
                self.settings.init_default,
                n_channels,
                groups=None if groups is None else [group.tolist() for group in groups],
                include_current=False,
                min_reref_size=MIN_REREF_GROUP_SIZE,
            )
            self._state.affine = AffineTransformTransformer(
                AffineTransformSettings(
                    weights=effective,
                    axis=self.settings.axis,
                    kernel=self.settings.kernel,
                )
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
    :class:`AxisArray` for self-supervised
    training via ``INPUT_SAMPLE``.
    """

    SETTINGS = LRRSettings

    INPUT_SAMPLE = ez.InputStream(AxisArray)

    @ez.subscriber(INPUT_SAMPLE)
    async def on_sample(self, msg: AxisArray) -> None:
        await self.processor.apartial_fit(msg)
