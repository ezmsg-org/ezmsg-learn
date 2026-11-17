"""A trained model resets when, and only when, something it depends on changes.

Per class: relabelled channels (same count) and a dtype change on a declared stream keep the model;
a feature-count change rebuilds it (or, where it cannot be rebuilt, raises a clear ValueError).
Also: what a segment change (a gap) does and does not reset.
"""

import pickle

import numpy as np
import pytest
from ezmsg.baseproc import SampleTriggerMessage
from ezmsg.util.messages.axisarray import AxisArray, compute_config_id, replace
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from ezmsg.learn.dim_reduce.adaptive_decomp import IncrementalPCASettings, IncrementalPCATransformer
from ezmsg.learn.process.linear_regressor import LinearRegressorTransformer
from ezmsg.learn.process.mlp_old import MLPProcessor, MLPSettings
from ezmsg.learn.process.refit_kalman import RefitKalmanFilterProcessor, RefitKalmanFilterSettings
from ezmsg.learn.process.rnn import RNNProcessor, RNNSettings
from ezmsg.learn.process.sgd import SGDDecoderSettings, SGDDecoderTransformer
from ezmsg.learn.process.sklearn import SklearnModelProcessor, SklearnModelSettings
from ezmsg.learn.process.slda import SLDASettings, SLDATransformer
from ezmsg.learn.process.ssr import LRRSettings, LRRTransformer
from ezmsg.learn.process.torch import (
    TorchModelProcessor,
    TorchModelSettings,
    TorchSimpleProcessor,
    TorchSimpleSettings,
)
from ezmsg.learn.process.transformer import TransformerProcessor, TransformerSettings

DUMMY = "tests.unit.test_torch.DummyModel"


def _msg(n_ch=4, prefix="c", dtype=np.float64, segment=0, key="sig", n_t=20, seed=0):
    msg = AxisArray(
        data=np.random.default_rng(seed).normal(size=(n_t, n_ch)).astype(dtype),
        dims=["time", "ch"],
        axes={
            "time": AxisArray.TimeAxis(fs=100.0),
            "ch": AxisArray.CoordinateAxis(data=np.array([f"{prefix}{i}" for i in range(n_ch)]), dims=["ch"]),
        },
        key=key,
        stream_dim="time",
        segment=segment,
    )
    return replace(msg, config_id=compute_config_id(msg))


def _trig(msg, value):
    return replace(msg, attrs={"trigger": SampleTriggerMessage(timestamp=0.0, value=value)})


def _call(proc, msg):
    proc(msg)


def _fit_call(proc, msg):
    proc.partial_fit(msg)
    proc(msg)


def _sklearn_fit_call(proc, msg):
    proc.partial_fit(_trig(msg, msg.data[:, 0]))
    proc(msg)


def _linreg_fit_call(proc, msg):
    target = AxisArray(
        data=msg.data[:, :2] @ np.ones((2, 2)),
        dims=["time", "ch"],
        axes={"ch": AxisArray.CoordinateAxis(data=np.array(["y0", "y1"]), dims=["ch"])},
        key="y",
    )
    proc.partial_fit(_trig(msg, target))
    proc(msg)


def _kalman(tmp_path):
    path = tmp_path / "kalman.pkl"
    with open(path, "wb") as f:
        pickle.dump(
            {
                "A_state_transition_matrix": np.array([[1.0, 0.1], [0.0, 1.0]]),
                "H_observation_matrix": np.arange(8.0).reshape(4, 2) / 8,
                "W_process_noise_covariance": np.eye(2) * 0.05,
                "Q_measurement_noise_covariance": np.eye(4) * 0.1,
            },
            f,
        )
    return RefitKalmanFilterProcessor(RefitKalmanFilterSettings(checkpoint_path=str(path), steady_state=False))


def _slda(tmp_path):
    lda = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
    lda.classes_ = np.array([0, 1])
    lda.coef_ = np.array([[0.5, -0.5, 0.2, 0.1]])
    lda.intercept_ = np.array([0.1])
    path = tmp_path / "lda.pkl"
    with open(path, "wb") as f:
        pickle.dump(lda, f)
    return SLDATransformer(SLDASettings(settings_path=str(path)))


# name -> (make(tmp_path), warm(proc, msg), the learned state, rebuilt on more features (else ValueError))
CASES = {
    "sklearn": (
        lambda _: SklearnModelProcessor(SklearnModelSettings(model_class="sklearn.linear_model.SGDRegressor")),
        _sklearn_fit_call,
        lambda p: p.state.model,
        True,
    ),
    "linear_regressor": (lambda _: LinearRegressorTransformer(), _linreg_fit_call, lambda p: p.state.model, True),
    "torch_simple": (
        lambda _: TorchSimpleProcessor(TorchSimpleSettings(model_class=DUMMY, device="cpu")),
        _call,
        lambda p: p.state.model,
        True,
    ),
    "torch_model": (
        lambda _: TorchModelProcessor(TorchModelSettings(model_class=DUMMY, device="cpu")),
        _call,
        lambda p: p.state.model,
        True,
    ),
    "rnn": (
        lambda _: RNNProcessor(RNNSettings(model_kwargs={"hidden_size": 8, "output_size": 2}, device="cpu")),
        _call,
        lambda p: p.state.model,
        True,
    ),
    "transformer": (
        lambda _: TransformerProcessor(
            TransformerSettings(
                device="cpu",
                model_kwargs={"hidden_size": 8, "encoder_layers": 1, "output_size": 2, "attention_heads": 2},
            )
        ),
        _call,
        lambda p: p.state.model,
        True,
    ),
    "mlp_old": (lambda _: MLPProcessor(MLPSettings(hidden_channels=[8, 2])), _call, lambda p: p.state.model, True),
    "ipca": (
        lambda _: IncrementalPCATransformer(IncrementalPCASettings(n_components=2)),
        _fit_call,
        lambda p: p.state.estimator,
        True,
    ),
    "lrr": (lambda _: LRRTransformer(LRRSettings()), _fit_call, lambda p: p.state.weights, True),
    "kalman": (_kalman, _call, lambda p: p.state.model, False),
    "slda": (_slda, _call, lambda p: p.state.lda, False),
}


@pytest.mark.parametrize("name", list(CASES))
def test_model_kept_on_relabel_and_dtype_reset_on_feature_count(name, tmp_path):
    make, warm, learned, rebuildable = CASES[name]
    proc = make(tmp_path)
    warm(proc, _msg())
    model = learned(proc)

    proc(_msg(prefix="x", key="other"))  # relabelled channels, other key: a new config_id
    assert learned(proc) is model
    # dtype: no class here keeps learned state in the input's dtype (each casts, or promotes on accumulation).
    proc(_msg(dtype=np.float32))
    assert learned(proc) is model

    wider = _msg(n_ch=6)
    if rebuildable:
        warm(proc, wider)  # rebuilt for 6 features and usable, training included
        assert learned(proc) is not model
    else:
        with pytest.raises(ValueError, match="expects 4"):
            proc(wider)


def test_torch_checkpoint_feature_mismatch_is_clear(tmp_path):
    proc = TorchModelProcessor(TorchModelSettings(model_class=DUMMY, device="cpu"))
    proc(_msg())
    proc.save_checkpoint(str(tmp_path / "ckpt.pt"))
    loaded = TorchModelProcessor(
        TorchModelSettings(model_class=DUMMY, device="cpu", checkpoint_path=str(tmp_path / "ckpt.pt"))
    )
    loaded(_msg())
    with pytest.raises(ValueError, match="expects 4 input features"):
        loaded(_msg(n_ch=6))


def _windows(n_ch=2, prefix="c", dtype=np.float64, segment=0):
    win = AxisArray(
        data=np.random.default_rng(1).normal(scale=0.05, size=(2, 3, n_ch)).astype(dtype),
        dims=["win", "time", "ch"],
        axes={
            "win": AxisArray.TimeAxis(fs=10.0),
            "ch": AxisArray.CoordinateAxis(data=np.array([f"{prefix}{i}" for i in range(n_ch)]), dims=["ch"]),
        },
        key="w",
        stream_dim="win",
        segment=segment,
    )
    return replace(win, config_id=compute_config_id(win))


def _samples(n_ch=2):
    rng = np.random.default_rng(2)
    out = []
    for label in ["A", "B"] * 5:
        data = rng.normal(scale=0.05, size=(3, n_ch))
        data[0 if label == "A" else 1, 0] += 1.0
        out.append(AxisArray(data=data, dims=["time", "ch"], attrs={"trigger": SampleTriggerMessage(value=label)}))
    return out


def test_sgd_alternation_relabel_dtype_and_feature_count():
    proc = SGDDecoderTransformer(SGDDecoderSettings(alpha=1e-3, label_weights={"A": 1.0, "B": 1.0}))
    samples = _samples()
    for sample in samples:
        proc.partial_fit(sample)
    model = proc.state.model

    # Training samples (time, ch) and inference windows (win, time, ch) count the same features per sample.
    for sample in samples:
        assert proc(_windows(segment=3)) is not None
        proc.partial_fit(sample)
    assert proc.state.model is model
    proc(_windows(prefix="x"))
    proc(_windows(dtype=np.float32))
    assert proc.state.model is model

    assert proc(_windows(n_ch=3)) is None  # rebuilt, not yet trained: no output rather than a crash
    assert proc.state.model is not model
    for sample in _samples(n_ch=3):
        proc.partial_fit(sample)
    assert proc(_windows(n_ch=3)).data.shape == (2, 2)


def test_kalman_segment_change_restarts_trajectory_keeps_model_and_refit_log(tmp_path):
    proc = _kalman(tmp_path)
    msg = _msg()
    proc._prepare_state(msg)
    model = proc.state.model
    P0 = np.array(model.P_state_covariance)
    proc._process(msg)
    proc.log_for_refit(msg)
    assert np.any(proc.state.x != 0)
    assert not np.allclose(model.P_state_covariance, P0)

    proc._prepare_state(replace(msg, segment=1))
    assert proc.state.model is model
    assert len(proc.state.buffer_neural) == 1 and len(proc.state.buffer_state) == 1
    np.testing.assert_array_equal(proc.state.x, 0)
    np.testing.assert_allclose(model.P_state_covariance, P0)


def _count_resets(proc):
    calls = []
    reset = proc._reset_state
    proc._reset_state = lambda message: (calls.append(1), reset(message))
    return calls


def test_lrr_gap_and_alternation_rebuild_nothing():
    proc = LRRTransformer(LRRSettings(incremental=False))
    proc.partial_fit(_msg(key="train", segment=7))
    proc(_msg())
    affine = proc.state.affine
    lrr_resets, affine_resets = _count_resets(proc), _count_resets(affine)

    for segment in range(1, 4):
        proc.partial_fit(_msg(key="train", segment=7 + segment, seed=segment))
        msg = _msg(segment=segment)
        out = proc(msg)
    assert lrr_resets == [] and affine_resets == []
    assert proc.state.affine is affine
    np.testing.assert_allclose(out.data, msg.data @ (np.eye(4) - proc.state.weights), atol=1e-10)


def test_lrr_relabel_after_refit_applies_latest_weights():
    # The inner affine resets itself on the new config_id; it must rebuild from the latest fit.
    proc = LRRTransformer(LRRSettings(incremental=False))
    proc.partial_fit(_msg(seed=1))
    proc(_msg())
    proc.partial_fit(_msg(seed=2))
    msg = _msg(prefix="x")
    np.testing.assert_allclose(proc(msg).data, msg.data @ (np.eye(4) - proc.state.weights), atol=1e-10)


def test_ipca_alternating_training_and_signal_streams_rebuild_nothing():
    proc = IncrementalPCATransformer(IncrementalPCASettings(n_components=2))
    proc.partial_fit(_msg(key="train", segment=5))
    proc(_msg())
    resets = _count_resets(proc)
    for segment in range(1, 4):
        proc.partial_fit(_msg(key="train", segment=5 + segment, seed=segment))
        out = proc(_msg(prefix="x", key="other", segment=segment))
    assert resets == []
    assert out.key == "other" and out.segment == 3
