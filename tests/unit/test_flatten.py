import numpy as np

from ezmsg.learn.process.flatten import FlattenTransformer
from ezmsg.util.messages.axisarray import AxisArray


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
    np.testing.assert_array_equal(result.axes["ch"].data, np.arange(12))
