"""Tests for ezmsg.learn.process.ssr (Linear Regression Rereferencing)."""

import tempfile

import numpy as np
import pytest
from ezmsg.util.messages.axisarray import AxisArray

from ezmsg.learn.process.ssr import LRRSettings, LRRTransformer

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


class TestProcessBeforeFitRaises:
    def test_process_before_fit_raises(self):
        """Calling process before fitting must raise RuntimeError."""
        msg = _make_axisarray(_random_data())
        proc = LRRTransformer(LRRSettings())
        with pytest.raises(RuntimeError, match="not been fitted"):
            proc.send(msg)


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


class TestFitTransform:
    def test_fit_transform(self):
        """fit_transform matches separate partial_fit + process."""
        rng = np.random.default_rng(10)
        X = _random_data(rng=rng)
        msg = _make_axisarray(X)

        proc1 = LRRTransformer(LRRSettings())
        out1 = proc1.fit_transform(msg)

        proc2 = LRRTransformer(LRRSettings())
        proc2.partial_fit(msg)
        out2 = proc2.send(msg)

        np.testing.assert_allclose(out1.data, out2.data, atol=1e-12)


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
