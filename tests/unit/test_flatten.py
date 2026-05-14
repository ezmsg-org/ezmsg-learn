import numpy as np
from ezmsg.util.messages.axisarray import AxisArray

from ezmsg.learn.process.flatten import FlattenSettings, FlattenTransformer


def test_flatten_transformer_preserves_window_axis_as_time():
    data = np.arange(2 * 3 * 4, dtype=float).reshape(2, 3, 4)
    message = AxisArray(
        data=data,
        dims=["win", "time", "ch"],
        axes={
            "win": AxisArray.TimeAxis(fs=10.0, offset=0.2),
            "time": AxisArray.TimeAxis(fs=100.0, offset=-0.02),
            "ch": AxisArray.CoordinateAxis(
                data=np.array(["c0", "c1", "c2", "c3"]),
                dims=["ch"],
            ),
        },
        key="signal",
    )

    proc = FlattenTransformer(preserve_axis="win", sample_axis="time", feature_axis="ch")
    result = proc(message)

    assert result.dims == ["time", "ch"]
    assert result.data.shape == (2, 12)
    np.testing.assert_array_equal(result.data, data.reshape(2, 12))
    assert result.axes["time"] == message.axes["win"]

    # Merged ``ch`` axis carries integer ``lag`` + string ``ch`` fields,
    # plus sigproc's auto-composed cartesian-product ``label``.
    ch_data = result.axes["ch"].data
    assert ch_data.dtype.names is not None
    assert "lag" in ch_data.dtype.names
    assert "ch" in ch_data.dtype.names
    assert "label" in ch_data.dtype.names

    np.testing.assert_array_equal(
        ch_data["lag"],
        np.asarray([2, 2, 2, 2, 1, 1, 1, 1, 0, 0, 0, 0], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        ch_data["ch"],
        np.asarray(["c0", "c1", "c2", "c3"] * 3),
    )
    np.testing.assert_array_equal(
        ch_data["label"],
        np.asarray(
            [
                "t-2/c0",
                "t-2/c1",
                "t-2/c2",
                "t-2/c3",
                "t-1/c0",
                "t-1/c1",
                "t-1/c2",
                "t-1/c3",
                "t-0/c0",
                "t-0/c1",
                "t-0/c2",
                "t-0/c3",
            ]
        ),
    )


def test_flatten_inner_transformer_built_once_per_shape() -> None:
    """The wrapper caches its inner sigproc transformer across messages.

    Building a fresh :class:`SigprocFlattenTransformer` per message would
    discard the inner's per-shape caches (permutation map, target shape,
    cartesian-product CoordinateAxis) and add allocation overhead on the
    hot path.  This test asserts that ``_state.inner`` is the *same*
    object across multiple calls with the same shape, and is replaced
    when the shape changes.
    """
    transformer = FlattenTransformer(
        settings=FlattenSettings(
            preserve_axis="win",
            sample_axis="time",
            feature_axis="ch",
        )
    )
    msg_a = AxisArray(
        data=np.arange(12).reshape(2, 3, 2),
        dims=["win", "time", "ch"],
        axes={
            "win": AxisArray.TimeAxis(fs=50.0, offset=0.02),
            "time": AxisArray.TimeAxis(fs=50.0, offset=-0.04),
            "ch": AxisArray.CoordinateAxis(
                data=np.asarray(["spk", "sbp"]),
                dims=["ch"],
            ),
        },
    )

    transformer(msg_a)
    inner_after_first = transformer._state.inner
    assert inner_after_first is not None

    # Second call with same shape — inner must be reused, not rebuilt.
    transformer(msg_a)
    assert transformer._state.inner is inner_after_first, (
        "Inner sigproc transformer should be cached across messages of "
        "the same shape; got a new object on the second call"
    )

    # Shape change — inner must be rebuilt.
    msg_b = AxisArray(
        data=np.arange(20).reshape(2, 5, 2),
        dims=["win", "time", "ch"],
        axes={
            "win": AxisArray.TimeAxis(fs=50.0, offset=0.02),
            "time": AxisArray.TimeAxis(fs=50.0, offset=-0.04),
            "ch": AxisArray.CoordinateAxis(
                data=np.asarray(["spk", "sbp"]),
                dims=["ch"],
            ),
        },
    )
    transformer(msg_b)
    assert transformer._state.inner is not inner_after_first, (
        "Inner sigproc transformer should be rebuilt when the input shape changes (new lag count → new lag axis)"
    )


def test_flatten_expands_windowed_feature_names_into_lag_struct_fields() -> None:
    transformer = FlattenTransformer(
        settings=FlattenSettings(
            preserve_axis="win",
            sample_axis="time",
            feature_axis="ch",
        )
    )
    message = AxisArray(
        data=np.arange(12).reshape(2, 3, 2),
        dims=["win", "time", "ch"],
        axes={
            "win": AxisArray.TimeAxis(fs=50.0, offset=0.02),
            "time": AxisArray.TimeAxis(fs=50.0, offset=-0.04),
            "ch": AxisArray.CoordinateAxis(
                data=np.asarray(["spk", "sbp"]),
                dims=["ch"],
            ),
        },
    )

    output = transformer(message)

    assert output.dims == ["time", "ch"]
    np.testing.assert_array_equal(
        output.data,
        np.asarray(
            [
                [0, 1, 2, 3, 4, 5],
                [6, 7, 8, 9, 10, 11],
            ]
        ),
    )
    ch_data = output.axes["ch"].data
    assert ch_data.dtype.names is not None
    np.testing.assert_array_equal(
        ch_data["lag"],
        np.asarray([2, 2, 1, 1, 0, 0], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        ch_data["ch"],
        np.asarray(["spk", "sbp"] * 3),
    )
    np.testing.assert_array_equal(
        ch_data["label"],
        np.asarray(
            [
                "t-2/spk",
                "t-2/sbp",
                "t-1/spk",
                "t-1/sbp",
                "t-0/spk",
                "t-0/sbp",
            ]
        ),
    )
