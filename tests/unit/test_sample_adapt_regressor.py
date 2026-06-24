import pytest

from ezmsg.learn.collection.sample_adapt_regressor import (
    SampleAdaptRegressor,
    SampleAdaptRegressorSettings,
    _model_backend,
)


def test_sample_adapt_regressor_uses_windowed_decode_branch_when_configured():
    collection = SampleAdaptRegressor(
        settings=SampleAdaptRegressorSettings(
            decode_window_dur=0.2,
            decode_window_shift=0.01,
        )
    )
    collection.configure()

    network = collection.network()

    assert (collection.INPUT_SIGNAL, collection.WINDOW.INPUT_SIGNAL) in network
    assert (collection.WINDOW.OUTPUT_SIGNAL, collection.FLATTEN.INPUT_SIGNAL) in network
    assert (collection.FLATTEN.OUTPUT_SIGNAL, collection.REGRESSOR.INPUT_SIGNAL) in network
    assert (collection.INPUT_SIGNAL, collection.REGRESSOR.INPUT_SIGNAL) not in network


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


def test_linear_backend_wires_sample_path_and_no_adapter():
    collection = SampleAdaptRegressor(
        settings=SampleAdaptRegressorSettings(model_type="linear")
    )
    collection.configure()

    network = collection.network()

    # Online-adaptation sample path is present for the linear engine.
    assert (collection.INPUT_TRIGGER, collection.SEQSEQSAMPLER.INPUT_TRIGGER) in network
    assert (
        collection.SEQSEQSAMPLER.OUTPUT_SAMPLE,
        collection.REGRESSOR.INPUT_SAMPLE,
    ) in network
    # Linear emits the canonical _pred contract directly; no adapter in the graph.
    assert (collection.REGRESSOR.OUTPUT_SIGNAL, collection.OUTPUT_SIGNAL) in network
    assert (collection.ADAPTER.OUTPUT_SIGNAL, collection.OUTPUT_SIGNAL) not in network


def test_torch_backend_wires_decode_only_through_adapter():
    collection = SampleAdaptRegressor(
        settings=SampleAdaptRegressorSettings(model_type="mlp")
    )
    collection.configure()

    network = collection.network()

    # Decode-only path: signal -> torch engine -> adapter -> output.
    assert (collection.INPUT_SIGNAL, collection.TORCH_REGRESSOR.INPUT_SIGNAL) in network
    assert (
        collection.TORCH_REGRESSOR.OUTPUT_SIGNAL,
        collection.ADAPTER.INPUT_SIGNAL,
    ) in network
    assert (collection.ADAPTER.OUTPUT_SIGNAL, collection.OUTPUT_SIGNAL) in network
    # Linear engine and its sample/adapt path stay inert.
    assert (
        collection.SEQSEQSAMPLER.OUTPUT_SAMPLE,
        collection.REGRESSOR.INPUT_SAMPLE,
    ) not in network
    assert (collection.REGRESSOR.OUTPUT_SIGNAL, collection.OUTPUT_SIGNAL) not in network


def test_kalman_backend_wires_decode_only_through_adapter():
    collection = SampleAdaptRegressor(
        settings=SampleAdaptRegressorSettings(model_type="kalman")
    )
    collection.configure()

    network = collection.network()

    assert (collection.INPUT_SIGNAL, collection.KALMAN_REGRESSOR.INPUT_SIGNAL) in network
    assert (
        collection.KALMAN_REGRESSOR.OUTPUT_SIGNAL,
        collection.ADAPTER.INPUT_SIGNAL,
    ) in network
    assert (collection.ADAPTER.OUTPUT_SIGNAL, collection.OUTPUT_SIGNAL) in network
    assert (collection.REGRESSOR.OUTPUT_SIGNAL, collection.OUTPUT_SIGNAL) not in network


def test_non_linear_backend_uses_windowed_decode_branch():
    collection = SampleAdaptRegressor(
        settings=SampleAdaptRegressorSettings(
            model_type="mlp",
            decode_window_dur=0.2,
            decode_window_shift=0.01,
        )
    )
    collection.configure()

    network = collection.network()

    # Windowing feeds the torch engine, not the (inert) linear regressor.
    assert (collection.INPUT_SIGNAL, collection.WINDOW.INPUT_SIGNAL) in network
    assert (
        collection.FLATTEN.OUTPUT_SIGNAL,
        collection.TORCH_REGRESSOR.INPUT_SIGNAL,
    ) in network
    assert (
        collection.FLATTEN.OUTPUT_SIGNAL,
        collection.REGRESSOR.INPUT_SIGNAL,
    ) not in network
