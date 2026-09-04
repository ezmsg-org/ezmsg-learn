"""Coordinate axes built here are handed downstream ready to use.

``CoordinateAxis.fingerprint`` is what every stateful consumer keys its cached
state on. It is computed on first access and cached on the instance, and the
cache pickles with the axis, so whoever touches it first pays and everyone
after gets it free.

In-process that first toucher is usually the next stateful node, which primes
the axis as a side effect of hashing it. The gap is the process boundary:
unpickling builds a *new* axis object per message, so an axis that left its
producing process cold is re-checksummed by the first consumer in every
receiving process, on every message. Priming at construction closes that, and
these tests are what would notice it stopping.
"""

import pickle

import numpy as np
import pytest
from ezmsg.util.messages.axisarray import AxisArray, CoordinateAxis

from ezmsg.learn.util import with_fingerprint


def signal(labels, n_time=32, fs=100.0, key="dev"):
    return AxisArray(
        np.random.default_rng(0).standard_normal((n_time, len(labels))),
        dims=["time", "ch"],
        axes={
            "time": AxisArray.TimeAxis(fs=fs),
            "ch": CoordinateAxis(data=np.array(labels), dims=["ch"]),
        },
        key=key,
        chunk_dim="time",
    )


def created_axes(source: AxisArray, result: AxisArray) -> dict:
    """Coordinate axes on *result* that are not objects *source* handed in.

    Identity, not equality: an axis that merely passed through was primed by
    whoever hashed it, which would mask a producer that primes nothing.
    """
    incoming = {id(a) for a in source.axes.values()}
    return {d: a for d, a in result.axes.items() if isinstance(a, CoordinateAxis) and id(a) not in incoming}


class TestTheHelper:
    def test_it_returns_the_same_axis(self):
        axis = CoordinateAxis(data=np.array(["a", "b"]), dims=["ch"])
        assert with_fingerprint(axis) is axis

    def test_it_is_idempotent(self):
        axis = with_fingerprint(CoordinateAxis(data=np.array(["a", "b"]), dims=["ch"]))
        first = axis.__dict__["_fingerprint"]
        assert with_fingerprint(axis).__dict__["_fingerprint"] is first

    @pytest.mark.parametrize("dtype", ["U8", "f8", "i4"])
    def test_priming_survives_the_transport(self, dtype):
        """The whole point: the far side gets the answer without recomputing."""
        axis = with_fingerprint(CoordinateAxis(data=np.arange(8).astype(dtype), dims=["ch"]))
        landed = pickle.loads(pickle.dumps(axis))
        assert "_fingerprint" in landed.__dict__
        assert landed.__dict__["_fingerprint"] == axis.fingerprint


class TestCreatedAxesArePrimed:
    """Messages here carry no ``chunk_dim``: released ezmsg-sigproc does not set
    it, so that is what these transformers actually receive today. It is why
    ``FlattenTransformer.STREAMING_DIMS`` names ``win`` -- the base class's
    ``("time",)`` fallback would exclude the lag dimension, which is the one
    thing the lag axis is sized by."""

    def test_the_lag_axis_flatten_builds(self):
        """The lag axis is what *this* package builds. The merged output axis is
        built by the inner ezmsg-sigproc transformer and primed there, not here."""
        from ezmsg.learn.process.flatten import FlattenSettings, FlattenTransformer

        proc = FlattenTransformer(FlattenSettings(preserve_axis="win", sample_axis="time", feature_axis="ch"))
        proc(
            AxisArray(
                np.arange(24).reshape(2, 3, 4).astype(float),
                dims=["win", "time", "ch"],
                axes={
                    "win": AxisArray.TimeAxis(fs=50.0),
                    "time": AxisArray.TimeAxis(fs=50.0),
                    "ch": CoordinateAxis(data=np.array(["a", "b", "c", "d"]), dims=["ch"]),
                },
                key="dev",
            )
        )
        lag_axis = proc._state.lag_axis
        assert lag_axis is not None, "expected the lag case to be detected"
        assert "_fingerprint" in lag_axis.__dict__

    def test_the_component_axis_incremental_pca_builds(self):
        from ezmsg.learn.dim_reduce.adaptive_decomp import (
            IncrementalPCASettings,
            IncrementalPCATransformer,
        )

        proc = IncrementalPCATransformer(IncrementalPCASettings(n_components=2))
        msg = signal(["c0", "c1", "c2"], n_time=64)
        proc.partial_fit(msg)
        out = proc(msg)
        assert out is not None
        cold = [d for d, a in out.axes.items() if isinstance(a, CoordinateAxis) and "_fingerprint" not in a.__dict__]
        assert not cold, f"handed downstream cold: {cold}"


class TestTheFlattenFallbackIsRight:
    """``STREAMING_DIMS`` decides which dimension is excluded when the producer
    is silent, and getting it wrong is not a small error."""

    @staticmethod
    def _msg(n_win, n_lag):
        return AxisArray(
            np.zeros((n_win, n_lag, 4)),
            dims=["win", "time", "ch"],
            axes={
                "win": AxisArray.TimeAxis(fs=50.0),
                "time": AxisArray.TimeAxis(fs=50.0),
                "ch": CoordinateAxis(data=np.array(["a", "b", "c", "d"]), dims=["ch"]),
            },
            key="dev",
        )

    @staticmethod
    def _proc():
        from ezmsg.learn.process.flatten import FlattenSettings, FlattenTransformer

        return FlattenTransformer(FlattenSettings(preserve_axis="win", sample_axis="time", feature_axis="ch"))

    def test_a_window_length_change_rebuilds(self):
        """The lag axis is sized by it."""
        proc = self._proc()
        proc(self._msg(n_win=2, n_lag=3))
        inner = proc._state.inner
        proc(self._msg(n_win=2, n_lag=5))
        assert proc._state.inner is not inner

    def test_a_window_count_change_does_not(self):
        """That is just how many windows arrived."""
        proc = self._proc()
        proc(self._msg(n_win=2, n_lag=3))
        inner = proc._state.inner
        proc(self._msg(n_win=7, n_lag=3))
        assert proc._state.inner is inner
