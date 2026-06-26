import numpy as np
import pytest
from ezmsg.sigproc.window import WindowSettings, WindowTransformer
from ezmsg.util.messages.axisarray import AxisArray

from ezmsg.learn.collection.sample_adapt_regressor import (
    DecodeOutputAdapterProcessor,
    SampleAdaptRegressorSettings,
    _build_regressor_unit,
    _model_backend,
    build_sample_adapt_regressor,
)
from ezmsg.learn.process.adaptive_linear_regressor import AdaptiveLinearRegressorUnit
from ezmsg.learn.process.flatten import FlattenSettings, FlattenTransformer
from ezmsg.learn.process.refit_kalman import RefitKalmanFilterUnit
from ezmsg.learn.process.torch import TorchModelUnit


def _build(**kwargs):
    """Build + configure a decode collection for the given settings."""
    collection = build_sample_adapt_regressor(SampleAdaptRegressorSettings(**kwargs))
    collection.configure()
    return collection


# --- backend routing ---------------------------------------------------------


@pytest.mark.parametrize(
    "model_type, expected",
    [
        ("linear", "linear"),
        ("logistic", "linear"),
        ("sgd", "linear"),
        ("par", "linear"),
        ("ridge", "linear"),
        ("mlp", "torch"),
        ("MLP", "torch"),
        ("kalman", "kalman"),
        ("Kalman", "kalman"),
    ],
)
def test_model_backend_routes_model_type_to_engine(model_type, expected):
    assert _model_backend(model_type) == expected


@pytest.mark.parametrize(
    "model_type, expected_backend, expected_unit",
    [
        ("linear", "linear", AdaptiveLinearRegressorUnit),
        ("mlp", "torch", TorchModelUnit),
        ("kalman", "kalman", RefitKalmanFilterUnit),
    ],
)
def test_build_regressor_unit_selects_engine(model_type, expected_backend, expected_unit):
    unit, backend = _build_regressor_unit(SampleAdaptRegressorSettings(model_type=model_type))
    assert backend == expected_backend
    assert isinstance(unit, expected_unit)


# --- collection topology -----------------------------------------------------


def test_linear_backend_wires_sample_path_and_no_adapter():
    collection = _build(model_type="linear")
    network = collection.network()

    # The factory builds only the units the linear engine uses.
    assert hasattr(collection, "RESAMPLE")
    assert hasattr(collection, "SEQSEQSAMPLER")
    assert not hasattr(collection, "ADAPTER")

    # Online-adaptation sample path is present for the linear engine.
    assert (collection.INPUT_TRIGGER, collection.SEQSEQSAMPLER.INPUT_TRIGGER) in network
    assert (
        collection.SEQSEQSAMPLER.OUTPUT_SAMPLE,
        collection.REGRESSOR.INPUT_SAMPLE,
    ) in network
    # Linear emits the canonical _pred contract directly; no adapter in the graph.
    assert (collection.REGRESSOR.OUTPUT_SIGNAL, collection.OUTPUT_SIGNAL) in network
    # No windowing by default: signal flows straight into the regressor.
    assert (collection.INPUT_SIGNAL, collection.REGRESSOR.INPUT_SIGNAL) in network


@pytest.mark.parametrize(
    "model_type, expected_unit",
    [("mlp", TorchModelUnit), ("kalman", RefitKalmanFilterUnit)],
)
def test_non_linear_backend_wires_decode_only_through_adapter(model_type, expected_unit):
    collection = _build(model_type=model_type)
    network = collection.network()

    # Only the chosen engine + adapter exist; no inert sample-path units.
    assert isinstance(collection.REGRESSOR, expected_unit)
    assert hasattr(collection, "ADAPTER")
    assert not hasattr(collection, "RESAMPLE")
    assert not hasattr(collection, "SEQSEQSAMPLER")

    # Decode-only path: signal -> engine -> adapter -> output.
    assert (collection.INPUT_SIGNAL, collection.REGRESSOR.INPUT_SIGNAL) in network
    assert (
        collection.REGRESSOR.OUTPUT_SIGNAL,
        collection.ADAPTER.INPUT_SIGNAL,
    ) in network
    assert (collection.ADAPTER.OUTPUT_SIGNAL, collection.OUTPUT_SIGNAL) in network


@pytest.mark.parametrize("model_type", ["linear", "mlp", "kalman"])
def test_windowed_decode_branch_when_configured(model_type):
    collection = _build(
        model_type=model_type,
        decode_window_dur=0.2,
        decode_window_shift=0.01,
    )
    network = collection.network()

    assert (collection.INPUT_SIGNAL, collection.WINDOW.INPUT_SIGNAL) in network
    assert (collection.WINDOW.OUTPUT_SIGNAL, collection.FLATTEN.INPUT_SIGNAL) in network
    assert (
        collection.FLATTEN.OUTPUT_SIGNAL,
        collection.REGRESSOR.INPUT_SIGNAL,
    ) in network
    # Windowing replaces the direct signal->regressor edge.
    assert (collection.INPUT_SIGNAL, collection.REGRESSOR.INPUT_SIGNAL) not in network


def test_non_windowed_backend_has_no_window_units():
    collection = _build(model_type="mlp")
    assert not hasattr(collection, "WINDOW")
    assert not hasattr(collection, "FLATTEN")


# --- decode output adapter ---------------------------------------------------


def _adapter_message(data, *, dims, with_time=True, key="dec"):
    axes = {}
    if with_time:
        axes["time"] = AxisArray.TimeAxis(fs=50.0)
    return AxisArray(data=np.asarray(data, dtype=float), dims=dims, axes=axes, key=key)


def test_adapter_normalizes_output_to_time_ch():
    # Kalman-style output: (time, state) with state_dim == len(output_labels).
    proc = DecodeOutputAdapterProcessor(output_labels=["vx", "vy"])
    message = _adapter_message(np.arange(8).reshape(4, 2), dims=["time", "state"], key="kf")

    result = proc(message)

    assert result.dims == ["time", "ch"]
    assert result.data.shape == (4, 2)
    assert list(result.get_axis("ch").data) == ["vx", "vy"]
    assert result.key == "kf_pred"


def test_adapter_requires_time_axis():
    proc = DecodeOutputAdapterProcessor(output_labels=["vx", "vy"])
    message = _adapter_message(np.arange(2).reshape(1, 2), dims=["win", "ch"], with_time=False)

    with pytest.raises(ValueError, match="time"):
        proc(message)


# --- windowed path integration ----------------------------------------------


def test_windowed_path_renames_win_to_time_and_feeds_adapter():
    """End-to-end check of the windowed mlp/kalman feature path.

    The adapter's ``time``-axis guard is only safe because Window + the
    learn-side Flatten rename the window axis (``win``) to ``time`` on output.
    This chains the real Window -> Flatten -> adapter processors with the exact
    settings ``configure()`` applies for the windowed path, so a future change
    to Flatten's ``sample_axis`` semantics would fail here instead of only
    surfacing at runtime. The torch/kalman engine in between preserves
    ``message.axes``, so feeding the flattened output straight to the adapter
    exercises the same time-axis plumbing.
    """
    fs = 100.0
    window_dur, window_shift = 0.2, 0.01
    n_time, n_ch = 60, 3
    sig = AxisArray(
        data=np.arange(n_time * n_ch, dtype=float).reshape(n_time, n_ch),
        dims=["time", "ch"],
        axes={
            "time": AxisArray.TimeAxis(fs=fs, offset=0.0),
            "ch": AxisArray.CoordinateAxis(data=np.array(["c0", "c1", "c2"]), dims=["ch"]),
        },
        key="neural",
    )

    # Settings mirror SampleAdaptRegressor.configure() for the windowed branch.
    windower = WindowTransformer(
        WindowSettings(
            axis="time",
            newaxis="win",
            window_dur=window_dur,
            window_shift=window_shift,
            zero_pad_until="none",
        )
    )
    flatten = FlattenTransformer(FlattenSettings(preserve_axis="win", sample_axis="time", feature_axis="ch"))
    adapter = DecodeOutputAdapterProcessor(output_labels=None)

    windowed = windower(sig)
    assert windowed.dims == ["win", "time", "ch"]

    flat = flatten(windowed)
    # The window axis is preserved but renamed to "time"; the inner lag dim and
    # channels fold into the feature axis.
    assert flat.dims == ["time", "ch"]
    assert "time" in flat.axes
    # The renamed axis carries the window-rate cadence (one sample per shift),
    # not the original 100 Hz sample rate.
    assert flat.axes["time"].gain == pytest.approx(window_shift)

    # The adapter accepts the windowed output (no raise) and emits the contract.
    result = adapter(flat)
    assert result.dims == ["time", "ch"]
    assert result.data.shape[0] == flat.data.shape[0]
    assert result.key == "neural_pred"
    assert result.axes["time"].gain == pytest.approx(window_shift)
