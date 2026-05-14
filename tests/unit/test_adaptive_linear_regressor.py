import numpy as np
import pytest
import sklearn.linear_model
from ezmsg.sigproc.window import WindowTransformer
from ezmsg.sigproc.window import WindowSettings
from ezmsg.baseproc import SampleTriggerMessage
from ezmsg.util.messages.axisarray import AxisArray, replace

from ezmsg.learn.process.adaptive_linear_regressor import AdaptiveLinearRegressorTransformer
from ezmsg.learn.process.flatten import FlattenTransformer


def _make_signal(n_times: int = 128, n_ch: int = 3, fs: float = 1000.0) -> tuple[np.ndarray, AxisArray]:
    X = np.arange(n_times * n_ch, dtype=float).reshape((n_times, n_ch))
    signal = AxisArray(
        data=X,
        dims=["time", "ch"],
        axes={
            "time": AxisArray.TimeAxis(fs=fs, offset=0.0),
            "ch": AxisArray.CoordinateAxis(data=np.array([f"X{idx}" for idx in range(n_ch)]), dims=["ch"]),
        },
        key="signal",
    )
    return X, signal


def _make_value(data: np.ndarray, gain: float, labels: list[str], key: str = "value") -> AxisArray:
    return AxisArray(
        data=data,
        dims=["time", "ch"],
        axes={
            "time": AxisArray.TimeAxis(fs=1.0 / gain, offset=0.0),
            "ch": AxisArray.CoordinateAxis(data=np.array(labels), dims=["ch"]),
        },
        key=key,
    )


def _make_sample(signal: AxisArray, value: AxisArray) -> AxisArray:
    gain = signal.axes["time"].gain
    dur = signal.data.shape[0] * gain
    trigger = SampleTriggerMessage(
        timestamp=0.0,
        period=(0.0, dur),
        value=value,
    )
    return replace(signal, attrs={"trigger": trigger})


@pytest.mark.parametrize("model_type", ["linear", "logistic", "sgd", "par"])
def test_adaptive_linear_regressor_single_output(model_type: str):
    X, signal = _make_signal()
    target_raw = X @ np.arange(1, X.shape[1] + 1, dtype=float) + (X.shape[1] + 1)
    if model_type == "logistic":
        target_raw = (target_raw > np.median(target_raw)).astype(int)
    value = _make_value(target_raw[:, None], gain=signal.axes["time"].gain, labels=["y"])

    proc = AdaptiveLinearRegressorTransformer(model_type=model_type)
    proc.partial_fit(_make_sample(signal, value))
    preds = proc(signal)

    assert isinstance(preds, AxisArray)
    assert preds.data.shape == (signal.data.shape[0], 1)


def test_adaptive_linear_regressor_multi_output_river_linear():
    X, signal = _make_signal()
    y = np.column_stack(
        [
            X @ np.array([1.0, 2.0, 3.0]) + 1.0,
            X @ np.array([-1.0, 0.5, 2.0]) - 3.0,
        ]
    )
    value = _make_value(y, gain=signal.axes["time"].gain, labels=["y0", "y1"])

    proc = AdaptiveLinearRegressorTransformer(model_type="linear")
    proc.partial_fit(_make_sample(signal, value))
    preds = proc(signal)

    assert isinstance(preds, AxisArray)
    assert preds.data.shape == (signal.data.shape[0], 2)


def test_adaptive_linear_regressor_rejects_changed_river_target_labels():
    X, signal = _make_signal()
    y = np.column_stack(
        [
            X @ np.array([1.0, 2.0, 3.0]) + 1.0,
            X @ np.array([-1.0, 0.5, 2.0]) - 3.0,
        ]
    )

    proc = AdaptiveLinearRegressorTransformer(model_type="linear")
    proc.partial_fit(_make_sample(signal, _make_value(y, gain=signal.axes["time"].gain, labels=["y0", "y1"])))

    changed_labels = _make_value(y, gain=signal.axes["time"].gain, labels=["y0", "y2"])
    with pytest.raises(ValueError, match="Target labels do not match model labels."):
        proc.partial_fit(_make_sample(signal, changed_labels))


def test_adaptive_linear_regressor_predicts_from_checkpoint_before_partial_fit(tmp_path):
    X, signal = _make_signal()
    y = X @ np.array([1.0, 2.0, 3.0]) + 1.0
    model = sklearn.linear_model.SGDRegressor(random_state=0, max_iter=1000, tol=1e-3)
    model.fit(X, y)

    checkpoint_path = tmp_path / "adaptive_linear_regressor.pkl"
    with checkpoint_path.open("wb") as f:
        import pickle

        pickle.dump(model, f)

    proc = AdaptiveLinearRegressorTransformer(model_type="sgd", settings_path=str(checkpoint_path))
    preds = proc(signal)

    assert isinstance(preds, AxisArray)
    assert preds.data.shape == (signal.data.shape[0], 1)
    assert preds.axes["ch"].data.tolist() == ["ch0"]


def test_adaptive_linear_regressor_predicts_from_windowed_flattened_checkpoint(tmp_path):
    X, signal = _make_signal(n_times=10, n_ch=2, fs=10.0)
    win_len = 3
    X_win = np.stack([X[idx : idx + win_len].reshape(-1) for idx in range(len(X) - win_len + 1)])
    y = (X_win @ np.arange(1, X_win.shape[1] + 1, dtype=float)).reshape(-1, 1)

    model = sklearn.linear_model.SGDRegressor(random_state=0, max_iter=1000, tol=1e-3)
    model.fit(X_win, y.ravel())

    checkpoint_path = tmp_path / "adaptive_linear_regressor_windowed.pkl"
    with checkpoint_path.open("wb") as f:
        import pickle

        pickle.dump(model, f)

    windowing = WindowTransformer(
        WindowSettings(
            axis="time",
            newaxis="win",
            window_dur=win_len * signal.axes["time"].gain,
            window_shift=signal.axes["time"].gain,
        )
    )
    flatten = FlattenTransformer(preserve_axis="win", sample_axis="time", feature_axis="ch")
    proc = AdaptiveLinearRegressorTransformer(model_type="sgd", settings_path=str(checkpoint_path))

    preds = proc(flatten(windowing(signal)))

    assert isinstance(preds, AxisArray)
    assert preds.dims == ["time", "ch"]
    assert preds.data.shape == (len(X) - win_len + 1, 1)
