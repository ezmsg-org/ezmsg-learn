"""Stream metadata contract: segment changes keep learned state, outputs carry segment and a derived
config_id, and empty (end-of-stream) messages pass through."""

import numpy as np
import pytest
from ezmsg.baseproc import SampleTriggerMessage, end_of_stream
from ezmsg.util.messages.axisarray import AxisArray, compute_config_id, replace

from ezmsg.learn.dim_reduce.adaptive_decomp import IncrementalPCASettings, IncrementalPCATransformer
from ezmsg.learn.dim_reduce.incremental_decomp import IncrementalDecompSettings, IncrementalDecompTransformer
from ezmsg.learn.process.adaptive_linear_regressor import (
    AdaptiveLinearRegressorSettings,
    AdaptiveLinearRegressorTransformer,
)
from ezmsg.learn.process.rnn import RNNProcessor, RNNSettings
from ezmsg.learn.process.seqseqsampler import SeqSeqSampler
from ezmsg.learn.process.sgd import SGDDecoderSettings, SGDDecoderTransformer
from ezmsg.learn.process.sklearn import SklearnModelProcessor, SklearnModelSettings
from ezmsg.learn.process.ssr import LRRSettings, LRRTransformer
from ezmsg.learn.process.torch import TorchModelProcessor, TorchModelSettings


def _signal(rng, n_t=50, n_ch=4, segment=0, key="sig"):
    msg = AxisArray(
        data=rng.normal(size=(n_t, n_ch)),
        dims=["time", "ch"],
        axes={
            "time": AxisArray.TimeAxis(fs=100.0),
            "ch": AxisArray.CoordinateAxis(data=np.array([f"c{i}" for i in range(n_ch)]), dims=["ch"]),
        },
        key=key,
        stream_dim="time",
        segment=segment,
    )
    return replace(msg, config_id=compute_config_id(msg))


def _sgd_data(rng):
    labels = {"A": 0, "B": 1}
    samples = []
    for label in ["A", "B"] * 5:
        data = rng.normal(scale=0.05, size=(3, 2))
        data[labels[label], 0] += 1.0
        samples.append(
            AxisArray(
                data=data, dims=["time", "ch"], attrs={"trigger": SampleTriggerMessage(timestamp=0.0, value=label)}
            )
        )
    win = AxisArray(
        data=rng.normal(scale=0.05, size=(2, 3, 2)),
        dims=["win", "time", "ch"],
        axes={"win": AxisArray.TimeAxis(fs=10.0)},
        key="w",
        stream_dim="win",
    )
    return samples, replace(win, config_id=compute_config_id(win))


def test_sgd_segment_change_keeps_model():
    rng = np.random.default_rng(0)
    samples, win = _sgd_data(rng)
    proc = SGDDecoderTransformer(SGDDecoderSettings(alpha=1e-3, label_weights={"A": 1.0, "B": 1.0}))
    for sample in samples:
        proc.partial_fit(sample)
    model = proc.state.model
    out0 = proc(win)
    out1 = proc(replace(win, segment=1))
    assert proc.state.model is model
    np.testing.assert_array_equal(out0.data, out1.data)
    # Training continues without needing `classes` again.
    proc.partial_fit(samples[0])

    assert out1.segment == 1 and out1.stream_dim == "win"
    assert out1.config_id not in (0, win.config_id)
    assert out0.config_id == out1.config_id


def test_lrr_segment_change_keeps_fit_and_applies_latest_weights():
    rng = np.random.default_rng(1)
    proc = LRRTransformer(LRRSettings(incremental=False))
    proc.partial_fit(_signal(rng))
    proc(_signal(rng))  # builds the affine from the first fit
    proc.partial_fit(_signal(rng))  # refit in place
    weights = proc.state.weights

    msg = _signal(rng, segment=1)
    out = proc(msg)
    assert proc.state.weights is weights
    np.testing.assert_allclose(out.data, msg.data @ (np.eye(4) - weights), atol=1e-10)
    assert out.segment == 1


def test_lrr_training_on_new_segment_keeps_accumulating():
    rng = np.random.default_rng(2)
    proc = LRRTransformer(LRRSettings())
    proc.partial_fit(_signal(rng))
    proc.partial_fit(_signal(rng, segment=1))
    assert proc.state.n_samples == 100


def test_sklearn_segment_change_keeps_model():
    rng = np.random.default_rng(3)
    proc = SklearnModelProcessor(
        SklearnModelSettings(model_class="sklearn.linear_model.SGDClassifier", partial_fit_classes=np.array([0, 1]))
    )
    msg = _signal(rng)
    sample = replace(msg, attrs={"trigger": SampleTriggerMessage(timestamp=0.0, value=rng.integers(0, 2, size=50))})
    proc.partial_fit_transform(sample)
    model = proc.state.model
    out = proc(_signal(rng, segment=1))
    assert proc.state.model is model
    assert out.segment == 1 and out.config_id not in (0, msg.config_id)


def test_torch_model_segment_change_keeps_weights():
    rng = np.random.default_rng(4)
    proc = TorchModelProcessor(TorchModelSettings(model_class="tests.unit.test_torch.DummyModel", device="cpu"))
    proc(_signal(rng))
    model = proc.state.model
    out = proc(_signal(rng, segment=1))[0]
    assert proc.state.model is model
    assert out.segment == 1 and out.config_id not in (0, _signal(rng).config_id)


def test_rnn_segment_change_resets_hidden_keeps_model():
    rng = np.random.default_rng(5)
    proc = RNNProcessor(
        RNNSettings(model_kwargs={"hidden_size": 8, "output_size": 2}, device="cpu", preserve_state_across_windows=True)
    )
    proc(_signal(rng))
    model = proc.state.model
    assert proc.state.hx.abs().sum() > 0
    proc._reset_segment(_signal(rng, segment=1))
    assert proc.state.model is model
    assert proc.state.hx.abs().sum() == 0


def test_alr_output_carries_segment_and_derived_config_id():
    rng = np.random.default_rng(6)
    proc = AdaptiveLinearRegressorTransformer(AdaptiveLinearRegressorSettings(model_type="linear"))
    msg = _signal(rng)
    target = _signal(rng, n_ch=2, key="y", segment=7)  # the label stream's segment must not leak
    proc.partial_fit(replace(msg, attrs={"trigger": SampleTriggerMessage(timestamp=0.0, value=target)}))
    out = proc(replace(msg, segment=3))
    assert out.segment == 3 and out.stream_dim == "time"
    assert out.config_id not in (0, msg.config_id, target.config_id)
    # An undeclared input gives an undeclared output.
    assert proc(replace(msg, config_id=0)).config_id == 0


def test_ipca_output_config_id_differs_from_input():
    rng = np.random.default_rng(7)
    proc = IncrementalPCATransformer(IncrementalPCASettings(n_components=2))
    msg = _signal(rng)
    proc.partial_fit(msg)
    out = proc(replace(msg, segment=2))
    assert out.data.shape == (50, 2)
    assert out.segment == 2
    assert out.config_id not in (0, msg.config_id)


def test_incremental_decomp_windowed_training_does_not_reset_decomp():
    rng = np.random.default_rng(8)
    proc = IncrementalDecompTransformer(IncrementalDecompSettings(n_components=2, update_interval=0.1))
    decomp = proc._procs["decomp"]
    resets = []
    reset = decomp._reset_state
    decomp._reset_state = lambda message: (resets.append(message.config_id), reset(message))
    for _ in range(5):
        proc(_signal(rng))
    assert len(resets) == 1


def test_seqseqsampler_restarts_buffer_on_new_segment():
    rng = np.random.default_rng(9)
    sampler = SeqSeqSampler()
    msg = _signal(rng)
    sampler._update_buffer(msg, "signal")
    sampler._update_buffer(end_of_stream(msg), "signal")  # empty: must not raise
    assert sampler._signal_buffer.data.shape[0] == 0
    sampler._update_buffer(_signal(rng, segment=2), "signal")
    assert sampler._signal_buffer.segment == 2
    assert sampler._signal_buffer.data.shape[0] == 50


def _trained(name, rng):
    msg = _signal(rng)
    if name == "alr":
        proc = AdaptiveLinearRegressorTransformer(AdaptiveLinearRegressorSettings(model_type="sgd"))
        target = _signal(rng, n_ch=1)
        proc.partial_fit(replace(msg, attrs={"trigger": SampleTriggerMessage(timestamp=0.0, value=target)}))
    elif name == "sklearn":
        proc = SklearnModelProcessor(SklearnModelSettings(model_class="sklearn.linear_model.SGDRegressor"))
        proc.partial_fit_transform(
            replace(msg, attrs={"trigger": SampleTriggerMessage(timestamp=0.0, value=rng.normal(size=50))})
        )
    elif name == "ipca":
        proc = IncrementalPCATransformer(IncrementalPCASettings(n_components=2))
        proc.partial_fit(msg)
    elif name == "lrr":
        proc = LRRTransformer(LRRSettings())
        proc.partial_fit(msg)
    elif name == "rnn":
        proc = RNNProcessor(RNNSettings(model_kwargs={"hidden_size": 8, "output_size": 2}, device="cpu"))
    return proc, msg


@pytest.mark.parametrize("name", ["alr", "sklearn", "ipca", "lrr", "rnn"])
def test_empty_message_passes(name):
    rng = np.random.default_rng(10)
    proc, msg = _trained(name, rng)
    proc(msg)
    out = proc(end_of_stream(msg))
    out = out[0] if isinstance(out, list) else out
    assert out.data.shape[0] == 0
    assert out.segment == msg.segment + 1
