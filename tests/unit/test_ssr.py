"""Tests for ezmsg.learn.process.ssr (Linear Regression Rereferencing)."""

import tempfile

import numpy as np
import pytest
from ezmsg.util.messages.axisarray import AxisArray

from ezmsg.learn.process.ssr import (
    MIN_REREF_CLUSTER_SIZE,
    LRRSettings,
    LRRTransformer,
    RereferenceInit,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_axisarray(
    data: np.ndarray,
    fs: float = 100.0,
    ch_axis: str = "ch",
    dims: list[str] | None = None,
    key: str = "test",
) -> AxisArray:
    """Create an AxisArray from 2-D (time x ch) data."""
    if dims is None:
        dims = ["time", ch_axis]
    axes = {"time": AxisArray.TimeAxis(fs=fs, offset=0.0)}
    return AxisArray(data=data, dims=dims, axes=axes, key=key)


def _random_data(n_times: int = 200, n_ch: int = 8, rng=None) -> np.ndarray:
    if rng is None:
        rng = np.random.default_rng(42)
    return rng.standard_normal((n_times, n_ch))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestFitThenProcessShape:
    def test_fit_then_process_shape(self):
        """Output shape must match input shape."""
        rng = np.random.default_rng(0)
        X = _random_data(rng=rng)
        msg = _make_axisarray(X)

        proc = LRRTransformer(LRRSettings())
        proc.partial_fit(msg)
        out = proc.send(msg)

        assert isinstance(out, AxisArray)
        assert out.data.shape == X.shape


class TestProcessBeforeFitPassthrough:
    def test_process_before_fit_passthrough(self):
        """Calling process before fitting should pass data through unchanged."""
        rng = np.random.default_rng(123)
        X = _random_data(rng=rng)
        msg = _make_axisarray(X)

        proc = LRRTransformer(LRRSettings())
        out = proc.send(msg)

        assert isinstance(out, AxisArray)
        assert out.data.shape == X.shape
        np.testing.assert_allclose(out.data, X, atol=1e-12)


class TestEffectiveWeightsIMinusW:
    def test_effective_weights_I_minus_W(self):
        """Output equals X @ (I - W) computed manually."""
        rng = np.random.default_rng(1)
        X = _random_data(n_times=300, n_ch=4, rng=rng)
        msg = _make_axisarray(X)

        proc = LRRTransformer(LRRSettings())
        proc.partial_fit(msg)
        out = proc.send(msg)

        W = proc.state.weights
        expected = X @ (np.eye(W.shape[0]) - W)
        np.testing.assert_allclose(out.data, expected, atol=1e-10)


class TestDiagonalZero:
    def test_diagonal_zero(self):
        """Diagonal of W must always be zero."""
        rng = np.random.default_rng(2)
        X = _random_data(rng=rng)
        msg = _make_axisarray(X)

        proc = LRRTransformer(LRRSettings())
        proc.partial_fit(msg)

        np.testing.assert_array_equal(np.diag(proc.state.weights), 0.0)


class TestChannelClusters:
    def test_channel_clusters(self):
        """Cross-cluster weights must be zero; within-cluster weights non-zero."""
        rng = np.random.default_rng(3)
        n_ch = 8
        clusters = [[0, 1, 2, 3], [4, 5, 6, 7]]
        X = _random_data(n_ch=n_ch, rng=rng)
        msg = _make_axisarray(X)

        proc = LRRTransformer(LRRSettings(channel_clusters=clusters))
        proc.partial_fit(msg)

        W = proc.state.weights

        # Cross-cluster should be zero
        for c1 in clusters:
            for c2 in clusters:
                if c1 is c2:
                    continue
                cross = W[np.ix_(c1, c2)]
                np.testing.assert_array_equal(cross, 0.0)

        # Within-cluster (off-diagonal) should be non-zero
        for cluster in clusters:
            sub = W[np.ix_(cluster, cluster)]
            off_diag = sub[~np.eye(len(cluster), dtype=bool)]
            assert np.any(off_diag != 0), "Expected non-zero within-cluster weights"


def _banked_axisarray(data: np.ndarray, banks: list[str], key: str = "test") -> AxisArray:
    """AxisArray whose ch axis is a structured CoordinateAxis with a bank field,
    like ezmsg-blackrock ChannelMap emits."""
    dt = np.dtype([("label", "U16"), ("bank", "U1"), ("elec", "i4")])
    ch = np.zeros(len(banks), dtype=dt)
    ch["bank"] = banks
    ch["elec"] = list(range(1, len(banks) + 1))
    ch["label"] = [f"ch{i}" for i in range(len(banks))]
    return AxisArray(
        data=data,
        dims=["time", "ch"],
        axes={"time": AxisArray.TimeAxis(fs=100.0, offset=0.0), "ch": AxisArray.CoordinateAxis(data=ch, dims=["ch"])},
        key=key,
    )


class TestClusterByField:
    def test_bank_field_matches_explicit_clusters(self):
        """cluster_by_field='bank' derives the same clusters (and weights) as
        passing the equivalent explicit channel_clusters."""
        rng = np.random.default_rng(7)
        banks = ["A", "A", "A", "A", "B", "B", "B", "B"]
        X = _random_data(n_ch=len(banks), rng=rng)

        proc_field = LRRTransformer(LRRSettings(axis="ch", cluster_by_field="bank"))
        proc_field.partial_fit(_banked_axisarray(X, banks))

        proc_explicit = LRRTransformer(LRRSettings(axis="ch", channel_clusters=[[0, 1, 2, 3], [4, 5, 6, 7]]))
        proc_explicit.partial_fit(_make_axisarray(X))

        np.testing.assert_array_equal(proc_field.state.weights, proc_explicit.state.weights)
        # And cross-bank weights are zero
        W = proc_field.state.weights
        np.testing.assert_array_equal(W[np.ix_([0, 1, 2, 3], [4, 5, 6, 7])], 0.0)

    def test_explicit_clusters_take_precedence(self):
        """Explicit channel_clusters win over cluster_by_field."""
        rng = np.random.default_rng(8)
        banks = ["A", "A", "A", "A", "B", "B", "B", "B"]
        X = _random_data(n_ch=len(banks), rng=rng)

        # One all-channel cluster should override the bank grouping.
        proc = LRRTransformer(LRRSettings(axis="ch", channel_clusters=[list(range(8))], cluster_by_field="bank"))
        proc.partial_fit(_banked_axisarray(X, banks))
        # With a single cluster, cross-"bank" weights are NOT forced to zero.
        W = proc.state.weights
        assert np.any(W[np.ix_([0, 1, 2, 3], [4, 5, 6, 7])] != 0)

    def test_missing_field_falls_back_to_block_size(self):
        """cluster_by_field with no structured bank field falls back to block_size."""
        rng = np.random.default_rng(9)
        n_ch = 8
        X = _random_data(n_ch=n_ch, rng=rng)
        # Plain axis (no structured bank field) + block_size=4 -> two contiguous blocks.
        proc_field = LRRTransformer(LRRSettings(axis="ch", cluster_by_field="bank", block_size=4))
        proc_field.partial_fit(_make_axisarray(X))

        proc_block = LRRTransformer(LRRSettings(axis="ch", block_size=4))
        proc_block.partial_fit(_make_axisarray(X))

        np.testing.assert_array_equal(proc_field.state.weights, proc_block.state.weights)

    def test_bank_field_value_change_is_not_detected(self):
        """Intentional concession (mirrors the ezmsg-sigproc CAR fix): a live bank
        remap at fixed key + channel count is NOT re-derived. ``_hash_message``
        folds only an O(1) "bank field present" boolean, not the field's bytes, so
        the per-message hash does not scale with channel count. A genuine remap on
        real hardware arrives with a new key or channel count (escape hatch below)."""
        rng = np.random.default_rng(11)
        X = _random_data(n_ch=4, rng=rng)
        proc = LRRTransformer(LRRSettings(axis="ch", cluster_by_field="bank"))

        # First arrangement: banks A,A,B,B -> clusters {0,1},{2,3}.
        proc.partial_fit(_banked_axisarray(X, ["A", "A", "B", "B"], key="x"))
        assert proc.state.resolved_clusters == [[0, 1], [2, 3]]
        np.testing.assert_array_equal(proc.state.weights[np.ix_([0, 1], [2, 3])], 0.0)

        # Same key + channel count, different banks -> hash unchanged, so the
        # cached clusters are (deliberately) NOT re-derived.
        proc.partial_fit(_banked_axisarray(X, ["A", "B", "A", "B"], key="x"))
        assert proc.state.resolved_clusters == [[0, 1], [2, 3]]

        # Escape hatch: a new key (as a real remap would carry) forces re-derivation.
        proc.partial_fit(_banked_axisarray(X, ["A", "B", "A", "B"], key="y"))
        assert proc.state.resolved_clusters == [[0, 2], [1, 3]]


class TestIncrementalAccumulates:
    def test_incremental_accumulates(self):
        """Two partial_fits with incremental=True should match one fit on concatenated data."""
        rng = np.random.default_rng(4)
        X1 = _random_data(n_times=100, rng=rng)
        X2 = _random_data(n_times=100, rng=rng)

        # Incremental: two calls
        proc_inc = LRRTransformer(LRRSettings(incremental=True))
        proc_inc.partial_fit(_make_axisarray(X1))
        proc_inc.partial_fit(_make_axisarray(X2))

        # Batch: one call on concatenated data
        proc_batch = LRRTransformer(LRRSettings(incremental=False))
        proc_batch.partial_fit(_make_axisarray(np.concatenate([X1, X2], axis=0)))

        np.testing.assert_allclose(proc_inc.state.weights, proc_batch.state.weights, atol=1e-10)


class TestBatchResetsEachCall:
    def test_batch_resets_each_call(self):
        """With incremental=False, the second partial_fit ignores the first."""
        rng = np.random.default_rng(5)
        X1 = _random_data(n_times=100, rng=rng)
        X2 = _random_data(n_times=100, rng=rng)

        # Non-incremental: two calls
        proc = LRRTransformer(LRRSettings(incremental=False))
        proc.partial_fit(_make_axisarray(X1))
        proc.partial_fit(_make_axisarray(X2))

        # Reference: single fit on X2 only
        proc_ref = LRRTransformer(LRRSettings(incremental=False))
        proc_ref.partial_fit(_make_axisarray(X2))

        np.testing.assert_allclose(proc.state.weights, proc_ref.state.weights, atol=1e-10)


class TestRidgeHandlesCollinearity:
    def test_ridge_handles_collinearity(self):
        """Identical channels should not crash when ridge_lambda > 0."""
        rng = np.random.default_rng(6)
        base = rng.standard_normal((200, 1))
        X = np.hstack([base, base, rng.standard_normal((200, 2))])
        msg = _make_axisarray(X)

        proc = LRRTransformer(LRRSettings(ridge_lambda=1.0))
        proc.partial_fit(msg)
        out = proc.send(msg)

        assert out.data.shape == X.shape
        assert np.all(np.isfinite(out.data))


class TestNanDataSkipped:
    def test_nan_data_skipped(self):
        """partial_fit with NaN data is a no-op."""
        rng = np.random.default_rng(7)
        X_good = _random_data(rng=rng)
        X_nan = _random_data(rng=rng)
        X_nan[0, 0] = np.nan

        proc = LRRTransformer(LRRSettings())
        proc.partial_fit(_make_axisarray(X_good))
        W_before = proc.state.weights.copy()

        proc.partial_fit(_make_axisarray(X_nan))
        np.testing.assert_array_equal(proc.state.weights, W_before)


class TestCustomAxisName:
    def test_custom_axis_name(self):
        """Works when the channel axis has a custom name like 'sensor'."""
        rng = np.random.default_rng(8)
        X = _random_data(n_ch=4, rng=rng)
        msg = AxisArray(
            data=X,
            dims=["time", "sensor"],
            axes={"time": AxisArray.TimeAxis(fs=100.0, offset=0.0)},
            key="test",
        )

        proc = LRRTransformer(LRRSettings(axis="sensor"))
        proc.partial_fit(msg)
        out = proc.send(msg)

        assert out.data.shape == X.shape


class TestNonLastAxis:
    def test_non_last_axis(self):
        """Channel axis in a middle position."""
        rng = np.random.default_rng(9)
        n_ch = 4
        # shape: (ch, time) — channels first
        X = rng.standard_normal((n_ch, 50))
        msg = AxisArray(
            data=X,
            dims=["ch", "time"],
            axes={"time": AxisArray.TimeAxis(fs=100.0, offset=0.0)},
            key="test",
        )

        proc = LRRTransformer(LRRSettings(axis="ch"))
        proc.partial_fit(msg)
        out = proc.send(msg)

        assert out.data.shape == X.shape


class TestPartialFitTransform:
    def test_partial_fit_transform(self):
        """partial_fit_transform matches separate partial_fit + process."""
        rng = np.random.default_rng(10)
        X = _random_data(rng=rng)
        msg = _make_axisarray(X)

        proc1 = LRRTransformer(LRRSettings())
        out1 = proc1.partial_fit_transform(msg)

        proc2 = LRRTransformer(LRRSettings())
        proc2.partial_fit(msg)
        out2 = proc2.send(msg)

        np.testing.assert_allclose(out1.data, out2.data, atol=1e-12)


class TestPassthroughThenFit:
    def test_passthrough_then_fit(self):
        """Pre-fit send() should passthrough, then partial_fit() should update weights."""
        rng = np.random.default_rng(124)
        X = _random_data(n_times=300, n_ch=4, rng=rng)
        msg = _make_axisarray(X)

        proc = LRRTransformer(LRRSettings())

        out_before = proc.send(msg)
        np.testing.assert_allclose(out_before.data, X, atol=1e-12)

        proc.partial_fit(msg)
        out_after = proc.send(msg)

        W = proc.state.weights
        expected = X @ (np.eye(W.shape[0]) - W)
        np.testing.assert_allclose(out_after.data, expected, atol=1e-10)


class TestInvalidClusterIndicesRaise:
    def test_invalid_cluster_indices_raise(self):
        """Out-of-range indices in channel_clusters should raise ValueError."""
        rng = np.random.default_rng(11)
        X = _random_data(n_ch=4, rng=rng)
        msg = _make_axisarray(X)

        proc = LRRTransformer(LRRSettings(channel_clusters=[[0, 1, 99]]))
        with pytest.raises(ValueError, match="out-of-range"):
            proc.partial_fit(msg)


class TestClustersEngageBlockDiagonal:
    def test_clusters_engage_block_diagonal(self):
        """When clusters create a block-diagonal I-W, AffineTransform uses cluster opt."""
        rng = np.random.default_rng(12)
        n_ch = 8
        clusters = [[0, 1, 2, 3], [4, 5, 6, 7]]
        X = _random_data(n_ch=n_ch, n_times=300, rng=rng)
        msg = _make_axisarray(X)

        proc = LRRTransformer(LRRSettings(channel_clusters=clusters, min_cluster_size=1))
        proc.partial_fit(msg)
        out = proc.send(msg)

        # Verify output is correct — the block-diagonal path should produce
        # the same result as a full matmul.
        W = proc.state.weights
        expected = X @ (np.eye(n_ch) - W)
        np.testing.assert_allclose(out.data, expected, atol=1e-10)


class TestPrecalculatedWeights:
    def test_precalculated_weights(self):
        """Pre-calculated weights skip fit and produce correct output."""
        rng = np.random.default_rng(13)
        n_ch = 4
        X = _random_data(n_ch=n_ch, rng=rng)

        # Fit once to get weights
        proc_fit = LRRTransformer(LRRSettings())
        proc_fit.partial_fit(_make_axisarray(X))
        W = proc_fit.state.weights.copy()

        # Use pre-calculated weights
        proc_pre = LRRTransformer(LRRSettings(weights=W))
        msg = _make_axisarray(X)
        out = proc_pre.send(msg)

        expected = X @ (np.eye(n_ch) - W)
        np.testing.assert_allclose(out.data, expected, atol=1e-10)


class TestPrecalculatedWeightsFromFile:
    def test_precalculated_weights_from_file(self):
        """Load pre-calculated weights from a CSV file."""
        rng = np.random.default_rng(14)
        n_ch = 4
        X = _random_data(n_ch=n_ch, rng=rng)

        # Fit once to get weights
        proc_fit = LRRTransformer(LRRSettings())
        proc_fit.partial_fit(_make_axisarray(X))
        W = proc_fit.state.weights.copy()

        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
            np.savetxt(f, W, delimiter=",")
            path = f.name

        proc_pre = LRRTransformer(LRRSettings(weights=path))
        msg = _make_axisarray(X)
        out = proc_pre.send(msg)

        expected = X @ (np.eye(n_ch) - W)
        np.testing.assert_allclose(out.data, expected, atol=1e-10)


def _common_mode_data(n_times: int = 400, n_ch: int = 8, rng=None) -> np.ndarray:
    """Noise plus a shared common-mode component, so rereferencing (which
    regresses out shared signal) produces a clearly non-identity output."""
    if rng is None:
        rng = np.random.default_rng(7)
    common = rng.standard_normal((n_times, 1))
    return rng.standard_normal((n_times, n_ch)) + common


class TestLowChannelPassthrough:
    """Clusters smaller than MIN_REREF_CLUSTER_SIZE (and empty inputs) pass
    through untouched instead of crashing -- so sliced/partial channel sets are
    safe (e.g. a hub left with no channels after an upstream region slice)."""

    def _fit_process(self, data: np.ndarray, banks: list[str]) -> np.ndarray:
        proc = LRRTransformer(LRRSettings(axis="ch", cluster_by_field="bank"))
        for _ in range(8):
            proc.partial_fit(_banked_axisarray(data, banks))
        return np.asarray(proc(_banked_axisarray(data, banks)).data)

    def test_zero_channels_passthrough(self):
        """0 channels (fully sliced-out hub) must not crash on fit or process."""
        proc = LRRTransformer(LRRSettings(axis="ch", cluster_by_field="bank"))
        empty = _banked_axisarray(np.zeros((10, 0)), [])
        proc.partial_fit(empty)  # no channels to fit -- must be a no-op
        out = proc(empty)  # must pass through, not build an affine from []
        assert out.data.shape == (10, 0)

    def test_zero_channels_batch_fit(self):
        """Batch fit() with 0 channels is the same no-op as partial_fit."""
        proc = LRRTransformer(LRRSettings(axis="ch"))
        proc.fit(np.zeros((10, 0)))
        out = proc(_banked_axisarray(np.zeros((10, 0)), []))
        assert out.data.shape == (10, 0)

    def test_single_channel_identity(self):
        rng = np.random.default_rng(1)
        X = _common_mode_data(n_ch=1, rng=rng)
        out = self._fit_process(X, ["A"])
        np.testing.assert_allclose(out, X, atol=1e-10)

    def test_below_threshold_identity(self):
        """A cluster with < MIN_REREF_CLUSTER_SIZE channels is left untouched."""
        n = MIN_REREF_CLUSTER_SIZE - 1
        rng = np.random.default_rng(2)
        X = _common_mode_data(n_ch=n, rng=rng)
        out = self._fit_process(X, ["A"] * n)
        np.testing.assert_allclose(out, X, atol=1e-10)

    def test_at_threshold_rereferences(self):
        """A cluster with exactly MIN_REREF_CLUSTER_SIZE channels is rereferenced."""
        n = MIN_REREF_CLUSTER_SIZE
        rng = np.random.default_rng(3)
        X = _common_mode_data(n_ch=n, rng=rng)
        out = self._fit_process(X, ["A"] * n)
        assert np.max(np.abs(out - X)) > 1e-3

    def test_mixed_small_and_large_clusters(self):
        """Per-cluster: a full bank rereferences while a lone-channel bank in the
        same message passes through untouched."""
        big = MIN_REREF_CLUSTER_SIZE + 1
        rng = np.random.default_rng(4)
        X = _common_mode_data(n_ch=big + 1, rng=rng)
        banks = ["A"] * big + ["B"]  # bank A: big ch, bank B: 1 ch
        out = self._fit_process(X, banks)
        np.testing.assert_allclose(out[:, big], X[:, big], atol=1e-10)  # lone B ch untouched
        assert np.max(np.abs(out[:, :big] - X[:, :big])) > 1e-3  # bank A rereferenced

    def test_empty_explicit_clusters_with_channels_raises(self):
        """channel_clusters=[] with real channels is a misconfiguration: fail fast
        rather than silently disable rereferencing (the empty list is only
        tolerated when there are no channels)."""
        proc = LRRTransformer(LRRSettings(axis="ch", channel_clusters=[]))
        with pytest.raises(ValueError, match="empty but the input has"):
            proc.partial_fit(_make_axisarray(_random_data(n_ch=8)))


class TestCARInit:
    """init_default=CAR: cold-start per-cluster leave-one-out CAR when there are
    no weights and nothing has been fit."""

    @staticmethod
    def _loo_car(X: np.ndarray, clusters) -> np.ndarray:
        """Reference per-cluster leave-one-out CAR: y_i = x_i - mean_{j!=i} x_j."""
        out = X.copy()
        for cl in clusters:
            if len(cl) < MIN_REREF_CLUSTER_SIZE:
                continue
            block = X[:, cl]
            loo = (block.sum(axis=1, keepdims=True) - block) / (len(cl) - 1)
            out[:, cl] = block - loo
        return out

    def test_car_applies_leave_one_out_per_cluster(self):
        clusters = [[0, 1, 2, 3], [4, 5, 6, 7]]
        X = _random_data(n_ch=8)
        proc = LRRTransformer(
            LRRSettings(channel_clusters=clusters, init_default=RereferenceInit.CAR)
        )
        out = proc.send(_make_axisarray(X))  # no fit / no weights
        np.testing.assert_allclose(out.data, self._loo_car(X, clusters), atol=1e-10)

    def test_car_leaves_small_clusters_identity(self):
        # first cluster (size 2 < MIN_REREF_CLUSTER_SIZE) must pass through
        clusters = [[0, 1], [2, 3, 4, 5, 6, 7]]
        X = _random_data(n_ch=8)
        proc = LRRTransformer(
            LRRSettings(channel_clusters=clusters, init_default=RereferenceInit.CAR)
        )
        out = proc.send(_make_axisarray(X))
        np.testing.assert_allclose(out.data[:, :2], X[:, :2], atol=1e-12)
        np.testing.assert_allclose(out.data, self._loo_car(X, clusters), atol=1e-10)

    def test_car_from_bank_field(self):
        """cluster_by_field='bank' + CAR reproduces per-bank leave-one-out CAR."""
        n_ch = 8
        ch = np.zeros(n_ch, dtype=[("bank", "U1")])
        ch["bank"][:4], ch["bank"][4:] = "A", "B"
        X = _random_data(n_ch=n_ch)
        msg = AxisArray(
            data=X,
            dims=["time", "ch"],
            axes={
                "time": AxisArray.TimeAxis(fs=100.0, offset=0.0),
                "ch": AxisArray.CoordinateAxis(data=ch, dims=["ch"]),
            },
            key="test",
        )
        proc = LRRTransformer(
            LRRSettings(axis="ch", cluster_by_field="bank", init_default=RereferenceInit.CAR)
        )
        out = proc.send(msg)
        np.testing.assert_allclose(
            out.data, self._loo_car(X, [[0, 1, 2, 3], [4, 5, 6, 7]]), atol=1e-10
        )

    def test_default_init_is_identity_passthrough(self):
        """Default (IDENTITY) with no weights is unchanged legacy passthrough."""
        X = _random_data(n_ch=8)
        proc = LRRTransformer(LRRSettings(channel_clusters=[[0, 1, 2, 3], [4, 5, 6, 7]]))
        out = proc.send(_make_axisarray(X))
        np.testing.assert_allclose(out.data, X, atol=1e-12)

    def test_provided_weights_override_car(self):
        """Explicit weights win over the CAR cold-start default."""
        X = _random_data(n_ch=8)
        # W = 0 => effective I - W = identity, so output is passthrough (not CAR).
        proc = LRRTransformer(
            LRRSettings(weights=np.zeros((8, 8)), init_default=RereferenceInit.CAR)
        )
        out = proc.send(_make_axisarray(X))
        np.testing.assert_allclose(out.data, X, atol=1e-12)

    def test_fit_overrides_car(self):
        """A fitted LRR takes precedence over the CAR cold-start default: once
        weights are learned, output is the fitted rereference, not CAR."""
        clusters = [[0, 1, 2, 3], [4, 5, 6, 7]]
        X = _random_data(n_ch=8, n_times=400)
        msg = _make_axisarray(X)
        proc = LRRTransformer(
            LRRSettings(channel_clusters=clusters, init_default=RereferenceInit.CAR)
        )
        proc.partial_fit(msg)
        out = proc.send(msg)

        fitted = X @ (np.eye(8) - proc.state.weights)
        np.testing.assert_allclose(out.data, fitted, atol=1e-8)
        # And it is NOT the CAR cold-start.
        assert not np.allclose(out.data, self._loo_car(X, clusters), atol=1e-8)
