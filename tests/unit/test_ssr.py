"""Tests for ezmsg.learn.process.ssr (Linear Regression Rereferencing)."""

import tempfile

import numpy as np
import pytest
from ezmsg.util.messages.axisarray import AxisArray

from ezmsg.learn.process.ssr import (
    MIN_REREF_GROUP_SIZE,
    LRRSettings,
    LRRTransformer,
    RereferenceKind,
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


def _common_mode_data(n_times: int = 400, n_ch: int = 8, rng=None) -> np.ndarray:
    """Noise plus a shared common-mode component, so rereferencing (which
    regresses out shared signal) produces a clearly non-identity output."""
    if rng is None:
        rng = np.random.default_rng(7)
    common = rng.standard_normal((n_times, 1))
    return rng.standard_normal((n_times, n_ch)) + common


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


class TestChannelGroups:
    def test_channel_groups(self):
        """Cross-group weights must be zero; within-group weights non-zero."""
        rng = np.random.default_rng(3)
        n_ch = 8
        groups = [[0, 1, 2, 3], [4, 5, 6, 7]]
        X = _random_data(n_ch=n_ch, rng=rng)
        msg = _make_axisarray(X)

        proc = LRRTransformer(LRRSettings(channel_groups=groups))
        proc.partial_fit(msg)

        W = proc.state.weights

        # Cross-group should be zero
        for c1 in groups:
            for c2 in groups:
                if c1 is c2:
                    continue
                cross = W[np.ix_(c1, c2)]
                np.testing.assert_array_equal(cross, 0.0)

        # Within-group (off-diagonal) should be non-zero
        for group in groups:
            sub = W[np.ix_(group, group)]
            off_diag = sub[~np.eye(len(group), dtype=bool)]
            assert np.any(off_diag != 0), "Expected non-zero within-group weights"


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


class TestGroupByField:
    def test_bank_field_matches_explicit_groups(self):
        """channel_groups='bank' derives the same groups (and weights) as
        passing the equivalent explicit channel_groups."""
        rng = np.random.default_rng(7)
        banks = ["A", "A", "A", "A", "B", "B", "B", "B"]
        X = _random_data(n_ch=len(banks), rng=rng)

        proc_field = LRRTransformer(LRRSettings(axis="ch", channel_groups="bank"))
        proc_field.partial_fit(_banked_axisarray(X, banks))

        proc_explicit = LRRTransformer(LRRSettings(axis="ch", channel_groups=[[0, 1, 2, 3], [4, 5, 6, 7]]))
        proc_explicit.partial_fit(_make_axisarray(X))

        np.testing.assert_array_equal(proc_field.state.weights, proc_explicit.state.weights)
        # And cross-bank weights are zero
        W = proc_field.state.weights
        np.testing.assert_array_equal(W[np.ix_([0, 1, 2, 3], [4, 5, 6, 7])], 0.0)

    def test_callable_spec(self):
        """A callable spec is resolved against the message like any other."""
        rng = np.random.default_rng(8)
        banks = ["A", "A", "A", "A", "B", "B", "B", "B"]
        X = _random_data(n_ch=len(banks), rng=rng)

        # A callable returning one all-channel group: cross-"bank" weights are
        # then NOT forced to zero.
        proc = LRRTransformer(LRRSettings(axis="ch", channel_groups=lambda msg, axis: [list(range(8))]))
        proc.partial_fit(_banked_axisarray(X, banks))
        W = proc.state.weights
        assert np.any(W[np.ix_([0, 1, 2, 3], [4, 5, 6, 7])] != 0)

    def test_multi_field_spec(self):
        """A tuple of field names groups by their combination."""
        n_ch = 8
        dt = np.dtype([("array", "U1"), ("bank", "U1")])
        ch = np.zeros(n_ch, dtype=dt)
        ch["array"] = ["1", "1", "1", "1", "2", "2", "2", "2"]
        ch["bank"] = ["A", "A", "B", "B", "A", "A", "B", "B"]
        X = _random_data(n_ch=n_ch, rng=np.random.default_rng(21))
        msg = AxisArray(
            data=X,
            dims=["time", "ch"],
            axes={
                "time": AxisArray.TimeAxis(fs=100.0, offset=0.0),
                "ch": AxisArray.CoordinateAxis(data=ch, dims=["ch"]),
            },
            key="test",
        )

        proc = LRRTransformer(LRRSettings(axis="ch", channel_groups=("array", "bank")))
        proc.partial_fit(msg)

        # (array, bank) -> four groups of 2, each below MIN_REREF_GROUP_SIZE, so
        # every weight stays zero (identity passthrough).
        assert [g.tolist() for g in proc.state.resolved_groups] == [[0, 1], [2, 3], [4, 5], [6, 7]]
        np.testing.assert_array_equal(proc.state.weights, 0.0)

    def test_missing_field_falls_back_to_block_size(self):
        """channel_groups with no structured bank field falls back to block_size."""
        rng = np.random.default_rng(9)
        n_ch = 8
        X = _random_data(n_ch=n_ch, rng=rng)
        # Plain axis (no structured bank field) + block_size=4 -> two contiguous blocks.
        proc_field = LRRTransformer(LRRSettings(axis="ch", channel_groups="bank", block_size=4))
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
        proc = LRRTransformer(LRRSettings(axis="ch", channel_groups="bank"))

        # First arrangement: banks A,A,B,B -> groups {0,1},{2,3}.
        proc.partial_fit(_banked_axisarray(X, ["A", "A", "B", "B"], key="x"))
        assert [g.tolist() for g in proc.state.resolved_groups] == [[0, 1], [2, 3]]
        np.testing.assert_array_equal(proc.state.weights[np.ix_([0, 1], [2, 3])], 0.0)

        # Same key + channel count, different banks -> hash unchanged, so the
        # cached groups are (deliberately) NOT re-derived.
        proc.partial_fit(_banked_axisarray(X, ["A", "B", "A", "B"], key="x"))
        assert [g.tolist() for g in proc.state.resolved_groups] == [[0, 1], [2, 3]]

        # Escape hatch: a new key (as a real remap would carry) forces re-derivation.
        proc.partial_fit(_banked_axisarray(X, ["A", "B", "A", "B"], key="y"))
        assert [g.tolist() for g in proc.state.resolved_groups] == [[0, 2], [1, 3]]


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


class TestRefitBeforeFirstMessage:
    """Regression: the internal affine must be built from the NEWEST weights.

    ``LRRUnit`` takes training on ``INPUT_SAMPLE`` and signal on a separate
    stream, so several ``partial_fit`` calls routinely land before the first
    message is processed. Building the affine eagerly in ``_on_weights_updated``
    put the first fit's weights in the affine's *settings*; its ``_reset_state``
    (deferred until a message arrives) then rebuilt from those settings and
    discarded every later fit, silently applying the first fit forever.
    """

    def test_multiple_fits_before_first_message(self):
        rng = np.random.default_rng(0)
        X1 = _random_data(n_ch=4, rng=rng)
        X2 = _random_data(n_ch=4, rng=rng)

        proc = LRRTransformer(LRRSettings(incremental=False))
        proc.partial_fit(_make_axisarray(X1))
        proc.partial_fit(_make_axisarray(X2))  # no message processed in between
        out = proc.send(_make_axisarray(X2))

        expected = X2 @ (np.eye(4) - proc.state.weights)
        np.testing.assert_allclose(out.data, expected, atol=1e-10)

        # And it is NOT the stale first fit.
        stale = LRRTransformer(LRRSettings(incremental=False))
        stale.partial_fit(_make_axisarray(X1))
        assert not np.allclose(out.data, X2 @ (np.eye(4) - stale.state.weights), atol=1e-8)

    def test_refit_after_first_message_still_applies(self):
        """The in-place set_weights path (affine already built) stays correct."""
        rng = np.random.default_rng(1)
        X1 = _random_data(n_ch=4, rng=rng)
        X2 = _random_data(n_ch=4, rng=rng)

        proc = LRRTransformer(LRRSettings(incremental=False))
        proc.partial_fit(_make_axisarray(X1))
        proc.send(_make_axisarray(X1))  # builds the affine
        proc.partial_fit(_make_axisarray(X2))
        out = proc.send(_make_axisarray(X2))

        expected = X2 @ (np.eye(4) - proc.state.weights)
        np.testing.assert_allclose(out.data, expected, atol=1e-10)


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


class TestInvalidGroupIndicesRaise:
    def test_invalid_group_indices_raise(self):
        """Out-of-range indices in channel_groups should raise ValueError."""
        rng = np.random.default_rng(11)
        X = _random_data(n_ch=4, rng=rng)
        msg = _make_axisarray(X)

        proc = LRRTransformer(LRRSettings(channel_groups=[[0, 1, 99]]))
        with pytest.raises(ValueError, match="out-of-range"):
            proc.partial_fit(msg)


class TestGroupsEngageBlockDiagonal:
    def test_groups_engage_block_diagonal(self):
        """kernel='blocks' forces the block-diagonal matmul over the blocks the
        affine reads off I - W; the result must equal the dense matmul."""
        rng = np.random.default_rng(12)
        n_ch = 8
        groups = [[0, 1, 2, 3], [4, 5, 6, 7]]
        X = _random_data(n_ch=n_ch, n_times=300, rng=rng)
        msg = _make_axisarray(X)

        proc = LRRTransformer(LRRSettings(channel_groups=groups, kernel="blocks"))
        proc.partial_fit(msg)
        out = proc.send(msg)

        # The affine really is on the block path, and it found the two 4-ch blocks.
        assert proc.state.affine.state.blocks is not None
        assert len(proc.state.affine.state.blocks) == 2

        # Verify output is correct — the block-diagonal path should produce
        # the same result as a full matmul.
        W = proc.state.weights
        expected = X @ (np.eye(n_ch) - W)
        np.testing.assert_allclose(out.data, expected, atol=1e-10)

    def test_kernel_dense_forces_dense(self):
        """kernel='dense' keeps the full matrix even when I - W is block-diagonal."""
        rng = np.random.default_rng(13)
        n_ch = 8
        X = _random_data(n_ch=n_ch, n_times=300, rng=rng)
        msg = _make_axisarray(X)

        proc = LRRTransformer(LRRSettings(channel_groups=[[0, 1, 2, 3], [4, 5, 6, 7]], kernel="dense"))
        proc.partial_fit(msg)
        out = proc.send(msg)

        assert proc.state.affine.state.blocks is None
        assert proc.state.affine.state.weights is not None
        expected = X @ (np.eye(n_ch) - proc.state.weights)
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


class TestApplyFollowsWeightBlocks:
    """Regression: applying fit/loaded weights must equal ``X @ (I - W)`` no matter
    what ``block_size`` / ``channel_groups`` the transformer carries.

    The block-diagonal apply optimization must follow the WEIGHT MATRIX's real
    block structure, not a group hint that may not match it. Previously the
    affine was built with ``_get_channel_groups()``, which falls back to
    ``block_size`` when the ``channel_groups`` metadata is absent at apply time
    (e.g. a processor constructed with weights before any message resolves the
    field). When those fallback groups were FINER than the W's true blocks --
    as when a W is fit over non-contiguous electrode-array groups but applied with
    a smaller ``block_size`` -- two input sub-groups of one true block mapped to
    the same output indices, and the block-diagonal matmul's assignment silently
    OVERWROTE the earlier sub-group. Each true block was then rereferenced against
    only a subset of its channels, corrupting the output while looking valid.

    ezmsg-sigproc#198 removed the hint's ability to matter at all: structure is
    now always read off W. This test keeps the guarantee pinned from this side.
    """

    def test_apply_ignores_block_size_finer_than_weight_blocks(self):
        rng = np.random.default_rng(123)
        n_ch = 128
        # W block-diagonal over two NON-CONTIGUOUS 64-ch groups (like two electrode
        # arrays interleaved in channel order: an "array" = two connector banks).
        groups = [list(range(0, 32)) + list(range(96, 128)), list(range(32, 96))]
        X = _random_data(n_times=600, n_ch=n_ch, rng=rng)

        proc_fit = LRRTransformer(LRRSettings(channel_groups=groups))
        proc_fit.partial_fit(_make_axisarray(X))
        W = proc_fit.state.weights.copy()
        # W really is block-diagonal over the non-contiguous 64-ch groups.
        for a in groups:
            for b in groups:
                if a is b:
                    continue
                np.testing.assert_array_equal(W[np.ix_(a, b)], 0.0)

        # Apply the loaded W with block_size=32 (finer than the W's 64-ch blocks).
        # These 4x32 groups do not match the W -> the block-diagonal apply used
        # to silently overwrite. The result must still be the faithful X @ (I - W).
        # kernel="blocks" forces the block path so the guarantee is tested there
        # rather than at whatever the planner happens to pick for this size.
        proc = LRRTransformer(LRRSettings(weights=W, block_size=32, kernel="blocks"))
        out = proc.send(_make_axisarray(X))
        expected = X @ (np.eye(n_ch) - W)
        np.testing.assert_allclose(out.data, expected, atol=1e-10)


class TestLowChannelPassthrough:
    """Groups smaller than MIN_REREF_GROUP_SIZE (and empty inputs) pass
    through untouched instead of crashing -- so sliced/partial channel sets are
    safe (e.g. a hub left with no channels after an upstream region slice)."""

    def _fit_process(self, data: np.ndarray, banks: list[str]) -> np.ndarray:
        proc = LRRTransformer(LRRSettings(axis="ch", channel_groups="bank"))
        for _ in range(8):
            proc.partial_fit(_banked_axisarray(data, banks))
        return np.asarray(proc(_banked_axisarray(data, banks)).data)

    def test_zero_channels_passthrough(self):
        """0 channels (fully sliced-out hub) must not crash on fit or process."""
        proc = LRRTransformer(LRRSettings(axis="ch", channel_groups="bank"))
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
        """A group with < MIN_REREF_GROUP_SIZE channels is left untouched."""
        n = MIN_REREF_GROUP_SIZE - 1
        rng = np.random.default_rng(2)
        X = _common_mode_data(n_ch=n, rng=rng)
        out = self._fit_process(X, ["A"] * n)
        np.testing.assert_allclose(out, X, atol=1e-10)

    def test_at_threshold_rereferences(self):
        """A group with exactly MIN_REREF_GROUP_SIZE channels is rereferenced."""
        n = MIN_REREF_GROUP_SIZE
        rng = np.random.default_rng(3)
        X = _common_mode_data(n_ch=n, rng=rng)
        out = self._fit_process(X, ["A"] * n)
        assert np.max(np.abs(out - X)) > 1e-3

    def test_mixed_small_and_large_groups(self):
        """Per-group: a full bank rereferences while a lone-channel bank in the
        same message passes through untouched."""
        big = MIN_REREF_GROUP_SIZE + 1
        rng = np.random.default_rng(4)
        X = _common_mode_data(n_ch=big + 1, rng=rng)
        banks = ["A"] * big + ["B"]  # bank A: big ch, bank B: 1 ch
        out = self._fit_process(X, banks)
        np.testing.assert_allclose(out[:, big], X[:, big], atol=1e-10)  # lone B ch untouched
        assert np.max(np.abs(out[:, :big] - X[:, :big])) > 1e-3  # bank A rereferenced

    def test_empty_explicit_groups_with_channels_raises(self):
        """channel_groups=[] with real channels is a misconfiguration: fail fast
        rather than silently disable rereferencing (the empty list is only
        tolerated when there are no channels)."""
        proc = LRRTransformer(LRRSettings(axis="ch", channel_groups=[]))
        with pytest.raises(ValueError, match="empty but the input has"):
            proc.partial_fit(_make_axisarray(_random_data(n_ch=8)))


class TestCARInit:
    """init_default=CAR: cold-start per-group leave-one-out CAR when there are
    no weights and nothing has been fit."""

    @staticmethod
    def _loo_car(X: np.ndarray, groups) -> np.ndarray:
        """Reference per-group leave-one-out CAR: y_i = x_i - mean_{j!=i} x_j."""
        out = X.copy()
        for cl in groups:
            if len(cl) < MIN_REREF_GROUP_SIZE:
                continue
            block = X[:, cl]
            loo = (block.sum(axis=1, keepdims=True) - block) / (len(cl) - 1)
            out[:, cl] = block - loo
        return out

    def test_car_applies_leave_one_out_per_group(self):
        groups = [[0, 1, 2, 3], [4, 5, 6, 7]]
        X = _random_data(n_ch=8)
        proc = LRRTransformer(LRRSettings(channel_groups=groups, init_default=RereferenceKind.CAR))
        out = proc.send(_make_axisarray(X))  # no fit / no weights
        np.testing.assert_allclose(out.data, self._loo_car(X, groups), atol=1e-10)

    def test_car_leaves_small_groups_identity(self):
        # first group (size 2 < MIN_REREF_GROUP_SIZE) must pass through
        groups = [[0, 1], [2, 3, 4, 5, 6, 7]]
        X = _random_data(n_ch=8)
        proc = LRRTransformer(LRRSettings(channel_groups=groups, init_default=RereferenceKind.CAR))
        out = proc.send(_make_axisarray(X))
        np.testing.assert_allclose(out.data[:, :2], X[:, :2], atol=1e-12)
        np.testing.assert_allclose(out.data, self._loo_car(X, groups), atol=1e-10)

    def test_car_from_bank_field(self):
        """channel_groups='bank' + CAR reproduces per-bank leave-one-out CAR."""
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
        proc = LRRTransformer(LRRSettings(axis="ch", channel_groups="bank", init_default=RereferenceKind.CAR))
        out = proc.send(msg)
        np.testing.assert_allclose(out.data, self._loo_car(X, [[0, 1, 2, 3], [4, 5, 6, 7]]), atol=1e-10)

    def test_default_init_is_identity_passthrough(self):
        """Default (IDENTITY) with no weights is unchanged legacy passthrough."""
        X = _random_data(n_ch=8)
        proc = LRRTransformer(LRRSettings(channel_groups=[[0, 1, 2, 3], [4, 5, 6, 7]]))
        out = proc.send(_make_axisarray(X))
        np.testing.assert_allclose(out.data, X, atol=1e-12)

    def test_provided_weights_override_car(self):
        """Explicit weights win over the CAR cold-start default."""
        X = _random_data(n_ch=8)
        # W = 0 => effective I - W = identity, so output is passthrough (not CAR).
        proc = LRRTransformer(LRRSettings(weights=np.zeros((8, 8)), init_default=RereferenceKind.CAR))
        out = proc.send(_make_axisarray(X))
        np.testing.assert_allclose(out.data, X, atol=1e-12)

    def test_fit_overrides_car(self):
        """A fitted LRR takes precedence over the CAR cold-start default: once
        weights are learned, output is the fitted rereference, not CAR."""
        groups = [[0, 1, 2, 3], [4, 5, 6, 7]]
        X = _random_data(n_ch=8, n_times=400)
        msg = _make_axisarray(X)
        proc = LRRTransformer(LRRSettings(channel_groups=groups, init_default=RereferenceKind.CAR))
        proc.partial_fit(msg)
        out = proc.send(msg)

        fitted = X @ (np.eye(8) - proc.state.weights)
        np.testing.assert_allclose(out.data, fitted, atol=1e-8)
        # And it is NOT the CAR cold-start.
        assert not np.allclose(out.data, self._loo_car(X, groups), atol=1e-8)


# ---------------------------------------------------------------------------
# Backend (array namespace) preservation
# ---------------------------------------------------------------------------


def _backend(name: str):
    """Return (converter, array_type) for a non-numpy Array API backend,
    skipping if the library is not installed (e.g. mlx off-macOS)."""
    if name == "mlx":
        mx = pytest.importorskip("mlx.core")
        return mx.array, mx.array
    torch = pytest.importorskip("torch")
    return torch.from_numpy, torch.Tensor


@pytest.mark.parametrize("backend", ["mlx", "torch"])
class TestBackendPreservation:
    """The input's array namespace (mlx / torch) must be preserved to the
    output, and derived state -- cxx, weights, and the internal affine's
    weight arrays -- must live in that namespace. Cold-start matrices are
    deliberately built as numpy and must be converted to the message's
    backend on first use by the affine transformer."""

    GROUPS = [[0, 1, 2, 3], [4, 5, 6, 7]]

    @staticmethod
    def _affine_weight_arrays(affine):
        """All weight arrays held by the internal affine (dense or per-group)."""
        if affine.state.weights is not None:
            return [affine.state.weights]
        return [sub_w for _, _, sub_w in affine.state.blocks]

    def test_cold_start_car_converts_and_preserves(self, backend):
        conv, typ = _backend(backend)
        X = _random_data().astype(np.float32)
        proc = LRRTransformer(LRRSettings(channel_groups=self.GROUPS, init_default=RereferenceKind.CAR))
        out = proc.send(_make_axisarray(conv(X.copy())))

        assert isinstance(out.data, typ)
        weight_arrays = self._affine_weight_arrays(proc.state.affine)
        assert len(weight_arrays) > 0
        for w in weight_arrays:
            assert isinstance(w, typ)

        # Values match the numpy cold-start CAR.
        ref_proc = LRRTransformer(LRRSettings(channel_groups=self.GROUPS, init_default=RereferenceKind.CAR))
        ref = ref_proc.send(_make_axisarray(X))
        np.testing.assert_allclose(np.asarray(out.data), ref.data, atol=1e-5)

    def test_fit_keeps_state_and_output_in_backend(self, backend):
        conv, typ = _backend(backend)
        X = _random_data(n_times=400).astype(np.float32)
        msg = _make_axisarray(conv(X.copy()))
        proc = LRRTransformer(LRRSettings(channel_groups=self.GROUPS))
        proc.partial_fit(msg)

        assert isinstance(proc.state.cxx, typ)
        assert isinstance(proc.state.weights, typ)

        out = proc.send(msg)
        assert isinstance(out.data, typ)
        for w in self._affine_weight_arrays(proc.state.affine):
            assert isinstance(w, typ)

        # Fitted output matches the numpy fit within float32 tolerance.
        ref_proc = LRRTransformer(LRRSettings(channel_groups=self.GROUPS))
        ref_proc.partial_fit(_make_axisarray(X))
        ref = ref_proc.send(_make_axisarray(X))
        np.testing.assert_allclose(np.asarray(out.data), ref.data, atol=1e-3)

    def test_numpy_settings_weights_with_backend_messages(self, backend):
        conv, typ = _backend(backend)
        X = _random_data(n_times=400).astype(np.float32)

        fit_proc = LRRTransformer(LRRSettings(channel_groups=self.GROUPS))
        fit_proc.partial_fit(_make_axisarray(X))
        W = np.asarray(fit_proc.state.weights)
        ref = fit_proc.send(_make_axisarray(X))

        proc = LRRTransformer(LRRSettings(weights=W, channel_groups=self.GROUPS))
        out = proc.send(_make_axisarray(conv(X.copy())))
        assert isinstance(out.data, typ)
        for w in self._affine_weight_arrays(proc.state.affine):
            assert isinstance(w, typ)
        np.testing.assert_allclose(np.asarray(out.data), ref.data, atol=1e-3)
