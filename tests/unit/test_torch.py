from pathlib import Path
import os
import sys

import numpy as np
import pytest
import torch
from ezmsg.sigproc.sampler import SampleMessage, SampleTriggerMessage
from ezmsg.util.messages.axisarray import AxisArray

from ezmsg.learn.process.torch import TorchModelProcessor

DUMMY_MODEL_CLASS = "tests.unit.test_torch.DummyModel"
MULTIHEAD_MODEL_CLASS = "tests.unit.test_torch.MultiHeadModel"


class DummyModel(torch.nn.Module):
    def __init__(self, input_size=4, output_size=2, dropout=0.0):
        super().__init__()
        self.linear = torch.nn.Linear(input_size, output_size)
        self.dropout = torch.nn.Dropout(dropout) if dropout > 0 else None

    def forward(self, x):
        if self.dropout:
            x = self.dropout(x)
        return self.linear(x)

    @classmethod
    def infer_config_from_state_dict(cls, state_dict):
        weight = (
            state_dict["linear.weight"]
            if isinstance(state_dict, dict)
            else state_dict.state_dict()["linear.weight"]
        )
        out_features, in_features = weight.shape
        return {
            "input_size": in_features,
            "output_size": out_features,
        }


class MultiHeadModel(torch.nn.Module):
    def __init__(self, input_size=4):
        super().__init__()
        self.head_a = torch.nn.Linear(input_size, 2)
        self.head_b = torch.nn.Linear(input_size, 3)

    def forward(self, x):
        return {
            "head_a": self.head_a(x),
            "head_b": self.head_b(x),
        }

    @classmethod
    def infer_config_from_state_dict(cls, state_dict):
        return {"input_size": state_dict["head_a.weight"].shape[1]}


@pytest.fixture
def device():
    """Returns 'cpu' if on macOS in GitHub Actions, otherwise None."""
    if os.getenv("GITHUB_ACTIONS") == "true" and sys.platform == "darwin":
        return "cpu"
    return None


@pytest.fixture(autouse=True)
def mps_memory_cleanup():
    """Fixture to clean up MPS memory after each test."""
    yield
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


@pytest.fixture
def batch_message():
    input_dim = 6
    batch_size = 10
    data = np.random.randn(batch_size, input_dim)
    return AxisArray(
        data=data,
        dims=["time", "ch"],
        axes={
            "time": AxisArray.TimeAxis(fs=100.0),
            "ch": AxisArray.CoordinateAxis(data=np.arange(input_dim), dims=["ch"]),
        },
    )


@pytest.mark.parametrize("input_size,output_size", [(4, 2), (6, 3), (8, 1)])
def test_inference_shapes(input_size, output_size, device):
    data = np.random.randn(12, input_size)
    msg = AxisArray(
        data=data,
        dims=["time", "ch"],
        axes={
            "time": AxisArray.TimeAxis(fs=100.0),
            "ch": AxisArray.CoordinateAxis(data=np.arange(input_size), dims=["ch"]),
        },
    )
    proc = TorchModelProcessor(
        model_class=DUMMY_MODEL_CLASS,
        model_kwargs={
            "input_size": input_size,
            "output_size": output_size,
        },
        device=device,
    )
    out = proc(msg)[0]
    # Check output last dim matches output_size
    assert out.data.shape[-1] == output_size
    # Check ch axis size
    assert out.get_axis("ch").data.shape[0] == output_size


def test_checkpoint_loading_and_weights(batch_message):
    proc = TorchModelProcessor(
        model_class=DUMMY_MODEL_CLASS,
        model_kwargs={
            "output_size": 2,
        },
        device="cpu",
    )
    proc(batch_message)  # initialize model

    checkpoint_filename = "test_torch_checkpoint.pth"
    proc.save_checkpoint(checkpoint_filename)

    proc2 = TorchModelProcessor(
        model_class=DUMMY_MODEL_CLASS,
        checkpoint_path=checkpoint_filename,
        device="cpu",
    )
    proc2(batch_message)

    model_state = proc._state.model.state_dict()
    loaded_state = proc2._state.model.state_dict()

    # Check all keys and values
    for key in model_state.keys():
        assert key in loaded_state, f"Key '{key}' missing in loaded model state"
        assert torch.allclose(model_state[key], loaded_state[key], atol=1e-6), (
            f"Mismatch for key '{key}'"
        )

    # Clean up checkpoint file
    for _ in range(5):
        Path(checkpoint_filename).unlink(missing_ok=True)


@pytest.mark.parametrize("dropout", [0.0, 0.1, 0.5])
def test_model_kwargs_propagation(dropout, batch_message, device):
    proc = TorchModelProcessor(
        model_class=DUMMY_MODEL_CLASS,
        model_kwargs={
            "output_size": 2,
            "dropout": dropout,
        },
        device=device,
    )
    proc(batch_message)
    model = proc._state.model
    if dropout > 0:
        assert isinstance(model.dropout, torch.nn.Dropout)
        assert model.dropout.p == dropout
    else:
        assert model.dropout is None


def test_partial_fit_changes_weights(batch_message, device):
    proc = TorchModelProcessor(
        model_class=DUMMY_MODEL_CLASS,
        loss_fn=torch.nn.MSELoss(),
        learning_rate=0.1,
        model_kwargs={
            "output_size": 2,
        },
        device=device,
    )
    x = batch_message.data[:1]
    y = np.random.randn(1, 2)

    sample = AxisArray(
        data=x,
        dims=["time", "ch"],
        axes={
            "time": AxisArray.TimeAxis(fs=100.0),
            "ch": AxisArray.CoordinateAxis(data=np.arange(x.shape[1]), dims=["ch"]),
        },
    )

    msg = SampleMessage(
        sample=sample,
        trigger=SampleTriggerMessage(timestamp=0.0, value=y),
    )

    proc(sample)  # run forward pass once to init model
    before = proc._state.model.linear.weight.clone()

    proc.partial_fit(msg)

    after = proc._state.model.linear.weight
    assert not torch.allclose(before, after)

    # Expect error if no loss function provided
    bad_proc = TorchModelProcessor(
        model_class=DUMMY_MODEL_CLASS,
        loss_fn=None,
        learning_rate=0.1,
        model_kwargs={
            "input_size": x.shape[-1],
            "output_size": 2,
        },
        device=device,
    )
    bad_proc(sample)
    with pytest.raises(ValueError):
        bad_proc.partial_fit(msg)


@pytest.mark.parametrize("device", ["cpu", "mps", "cuda"])
def test_model_runs_on_devices(device, batch_message):
    # Skip unavailable devices
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    if device == "mps":
        if not torch.backends.mps.is_available():
            pytest.skip("MPS not available")
        if os.getenv("GITHUB_ACTIONS") == "true":
            pytest.skip("MPS memory limit too low on free GitHub Actions runner")

    proc = TorchModelProcessor(
        model_class=DUMMY_MODEL_CLASS,
        device=device,
        model_kwargs={
            "output_size": 2,
        },
    )
    proc(batch_message)
    model = proc._state.model
    for param in model.parameters():
        assert param.device.type == device


@pytest.mark.parametrize("batch_size", [1, 5, 10])
def test_batch_processing(batch_size, device):
    input_dim = 4
    output_dim = 2
    data = np.random.randn(batch_size, input_dim)

    msg = AxisArray(
        data=data,
        dims=["time", "ch"],
        axes={
            "time": AxisArray.TimeAxis(fs=100.0),
            "ch": AxisArray.CoordinateAxis(data=np.arange(input_dim), dims=["ch"]),
        },
    )
    proc = TorchModelProcessor(
        model_class=DUMMY_MODEL_CLASS,
        model_kwargs={
            "input_size": input_dim,
            "output_size": output_dim,
        },
        device=device,
    )
    out = proc(msg)[0]
    assert out.data.shape[0] == batch_size
    assert out.data.shape[-1] == output_dim


def test_input_size_mismatch_raises_error():
    input_dim = 6
    wrong_input_dim = 4
    data = np.random.randn(10, input_dim)
    msg = AxisArray(
        data=data,
        dims=["time", "ch"],
        axes={
            "time": AxisArray.TimeAxis(fs=100.0),
            "ch": AxisArray.CoordinateAxis(data=np.arange(input_dim), dims=["ch"]),
        },
    )
    with pytest.raises(ValueError, match="Mismatch.*input_size.*"):
        TorchModelProcessor(
            model_class=DUMMY_MODEL_CLASS,
            model_kwargs={
                "input_size": wrong_input_dim,
                "output_size": 2,
            },
        )(msg)


def test_multihead_output(batch_message, device):
    proc = TorchModelProcessor(
        model_class=MULTIHEAD_MODEL_CLASS,
        model_kwargs={"input_size": batch_message.data.shape[1]},
        device=device,
    )
    results = proc(batch_message)

    keys = {r.key for r in results}
    assert keys == {"head_a", "head_b"}
    for r in results:
        assert r.data.ndim == 2


def test_multihead_partial_fit_with_loss_dict(batch_message, device):
    proc = TorchModelProcessor(
        model_class=MULTIHEAD_MODEL_CLASS,
        loss_fn={
            "head_a": torch.nn.MSELoss(),
            "head_b": torch.nn.L1Loss(),
        },
        model_kwargs={"input_size": batch_message.data.shape[1]},
        device=device,
    )

    proc(batch_message)  # initialize model

    y_targ = {
        "head_a": np.random.randn(1, 2),
        "head_b": np.random.randn(1, 3),
    }
    sample = AxisArray(
        data=batch_message.data[:1],
        dims=["time", "ch"],
        axes=batch_message.axes,
    )
    msg = SampleMessage(
        sample=sample,
        trigger=SampleTriggerMessage(timestamp=0.0, value=y_targ),
    )

    before_a = proc._state.model.head_a.weight.clone()
    before_b = proc._state.model.head_b.weight.clone()

    proc.partial_fit(msg)

    after_a = proc._state.model.head_a.weight
    after_b = proc._state.model.head_b.weight

    assert not torch.allclose(before_a, after_a)
    assert not torch.allclose(before_b, after_b)


def test_partial_fit_with_loss_weights(batch_message, device):
    proc = TorchModelProcessor(
        model_class=MULTIHEAD_MODEL_CLASS,
        loss_fn={
            "head_a": torch.nn.MSELoss(),
            "head_b": torch.nn.MSELoss(),
        },
        loss_weights={
            "head_a": 2.0,
            "head_b": 0.5,
        },
        model_kwargs={"input_size": batch_message.data.shape[1]},
        device=device,
    )
    proc(batch_message)

    y_targ = {
        "head_a": np.random.randn(1, 2),
        "head_b": np.random.randn(1, 3),
    }
    sample = AxisArray(
        data=batch_message.data[:1],
        dims=["time", "ch"],
        axes=batch_message.axes,
    )
    msg = SampleMessage(
        sample=sample,
        trigger=SampleTriggerMessage(timestamp=0.0, value=y_targ),
    )

    # Expect no error, and just run once
    proc.partial_fit(msg)
