"""Flatten with time-lag-windowing struct semantics.

Thin wrapper around :class:`ezmsg.sigproc.flatten.FlattenTransformer`
that detects the windowed-feature case (``(win, time, ch)`` 3-D input)
and attaches a structured ``lag`` :class:`CoordinateAxis` to the inner
sample dim before delegating.  The output merged-axis struct then
carries a real integer ``lag`` field alongside ``ch``, and sigproc's
canonical ``label`` field composes naturally (e.g. ``"t-2/c0"``).

Outside the lag case this module delegates unchanged — prefer
:class:`ezmsg.sigproc.flatten.Flatten` directly for the general
``(time, ch, feature) → (time, ch_x_feature)`` collapse.
"""

from __future__ import annotations

import typing

import ezmsg.core as ez
import numpy as np
from ezmsg.baseproc import (
    BaseStatefulTransformer,
    BaseTransformerUnit,
    processor_state,
)
from ezmsg.sigproc.flatten import (
    FlattenSettings as SigprocFlattenSettings,
)
from ezmsg.sigproc.flatten import (
    FlattenTransformer as SigprocFlattenTransformer,
)
from ezmsg.util.messages.axisarray import AxisArray, CoordinateAxis, replace


class FlattenSettings(ez.Settings):
    """Settings for the learn-side :obj:`Flatten`.

    Mirrors :class:`ezmsg.sigproc.flatten.FlattenSettings` but uses the
    historical ``feature_axis`` field name and defaults oriented toward
    time-lag windowing.
    """

    preserve_axis: str | None = None
    """Axis kept as the leading dim of the output.  Defaults to the
        input's leading dim (typically ``"win"`` in the windowed
        pipeline)."""

    sample_axis: str | None = None
    """Output name for the preserved axis.  Defaults to
        ``preserve_axis`` (no rename)."""

    feature_axis: str = "ch"
    """Output name for the merged axis."""


@processor_state
class _LagFlattenState:
    inner: typing.Any = None  # SigprocFlattenTransformer | None
    sample_dim: str = ""
    lag_axis: CoordinateAxis | None = None  # None outside the lag case


def _lag_sample_dim(
    message: AxisArray,
    preserve_axis: str,
    feature_axis: str,
) -> str | None:
    """Return the inner sample-dim name if the input matches
    ``(preserve, sample, feature)`` with a labeled feature axis; else None.
    """
    if (
        message.data.ndim != 3
        or message.dims[0] != preserve_axis
        or feature_axis not in message.dims
        or feature_axis not in message.axes
        or not hasattr(message.axes[feature_axis], "data")
    ):
        return None
    return next(
        (d for d in message.dims if d not in (preserve_axis, feature_axis)),
        None,
    )


def _build_lag_axis(sample_dim: str, sample_size: int) -> CoordinateAxis:
    """Structured CoordinateAxis carrying integer ``lag`` + ``label`` fields.

    Position 0 in the source time dim is the oldest sample → largest
    lag (``sample_size - 1``); position ``sample_size - 1`` is the most
    recent → ``lag = 0``.  The ``label`` sub-field is what sigproc's
    cartesian-product machinery picks as the primary, so the output
    merged-axis ``label`` reads e.g. ``"t-2/c0"``.
    """
    lags = np.arange(sample_size - 1, -1, -1, dtype=np.int32)
    label_strs = np.asarray([f"t-{i}" for i in lags])
    dtype = np.dtype([("lag", np.int32), ("label", label_strs.dtype)])
    data = np.empty(sample_size, dtype=dtype)
    data["lag"] = lags
    data["label"] = label_strs
    return CoordinateAxis(data=data, dims=[sample_dim])


class FlattenTransformer(BaseStatefulTransformer[FlattenSettings, AxisArray, AxisArray, _LagFlattenState]):
    """Time-lag-aware Flatten that delegates to the canonical sigproc unit.

    On each shape change we detect the lag case and, if present,
    precompute a structured lag :class:`CoordinateAxis`.  Per message
    that axis is injected into the inner sample dim before delegating
    to the inner sigproc transformer, which then produces an output
    merged-axis struct with ``lag`` (int), ``ch``, and the
    sigproc-composed ``"label"`` (``"t-2/c0"`` style).
    """

    def _hash_message(self, message: AxisArray) -> int:
        return hash((tuple(message.dims), tuple(message.data.shape)))

    def _reset_state(self, message: AxisArray) -> None:
        preserve_axis = self.settings.preserve_axis or message.dims[0]
        sample_axis = self.settings.sample_axis or preserve_axis
        feature_axis = self.settings.feature_axis

        if sample_axis == feature_axis:
            raise ValueError("sample_axis and feature_axis must be different.")

        sample_dim = _lag_sample_dim(message, preserve_axis, feature_axis)

        flatten_axes: tuple[str, ...] | None = None
        lag_axis: CoordinateAxis | None = None
        if sample_dim is not None:
            sample_size = message.data.shape[message.dims.index(sample_dim)]
            lag_axis = _build_lag_axis(sample_dim, sample_size)
            # Lock in (sample-slow, feature-fast) ordering so the
            # cartesian-product expansion matches the C-order reshape
            # (oldest lag first within each window).
            flatten_axes = (sample_dim, feature_axis)

        self._state.inner = SigprocFlattenTransformer(
            SigprocFlattenSettings(
                preserve_axis=preserve_axis,
                sample_axis=sample_axis,
                flatten_axes=flatten_axes,
                output_axis=feature_axis,
            )
        )
        self._state.sample_dim = sample_dim or ""
        self._state.lag_axis = lag_axis

    def _process(self, message: AxisArray) -> AxisArray:
        if self._state.lag_axis is not None:
            new_axes = dict(message.axes)
            new_axes[self._state.sample_dim] = self._state.lag_axis
            message = replace(message, axes=new_axes)
        return self._state.inner(message)


class Flatten(BaseTransformerUnit[FlattenSettings, AxisArray, AxisArray, FlattenTransformer]):
    SETTINGS = FlattenSettings
