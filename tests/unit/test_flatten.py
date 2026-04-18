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
    np.testing.assert_array_equal(
        result.axes["ch"].data,
        np.asarray(
            [
                "c0__t-2",
                "c1__t-2",
                "c2__t-2",
                "c3__t-2",
                "c0__t-1",
                "c1__t-1",
                "c2__t-1",
                "c3__t-1",
                "c0__t-0",
                "c1__t-0",
                "c2__t-0",
                "c3__t-0",
            ]
        ),
    )


def test_flatten_expands_windowed_feature_names_into_training_style_lags() -> None:
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
    np.testing.assert_array_equal(
        output.axes["ch"].data,
        np.asarray(
            [
                "spk__t-2",
                "sbp__t-2",
                "spk__t-1",
                "sbp__t-1",
                "spk__t-0",
                "sbp__t-0",
            ]
        ),
    )
