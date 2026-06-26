import numpy as np
import pytest
from ezmsg.util.messages.axisarray import AxisArray

from ezmsg.learn.collection.sample_adapt_regressor import (
    DecodeOutputAdapterProcessor,
    SampleAdaptRegressorSettings,
    _build_regressor_unit,
    _model_backend,
    build_sample_adapt_regressor,
)
from ezmsg.learn.process.adaptive_linear_regressor import AdaptiveLinearRegressorUnit
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
    message = _adapter_message(
        np.arange(8).reshape(4, 2), dims=["time", "state"], key="kf"
    )

    result = proc(message)

    assert result.dims == ["time", "ch"]
    assert result.data.shape == (4, 2)
    assert list(result.get_axis("ch").data) == ["vx", "vy"]
    assert result.key == "kf_pred"


def test_adapter_requires_time_axis():
    proc = DecodeOutputAdapterProcessor(output_labels=["vx", "vy"])
    message = _adapter_message(
        np.arange(2).reshape(1, 2), dims=["win", "ch"], with_time=False
    )

    with pytest.raises(ValueError, match="time"):
        proc(message)
