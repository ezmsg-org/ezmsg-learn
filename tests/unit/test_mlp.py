import numpy as np
import pytest
import torch
from ezmsg.util.messages.axisarray import AxisArray

from ezmsg.learn.process.torch import TorchModelProcessor


@pytest.fixture
def mlp_settings():
    return {
        "input_size": 8,
        "hidden_size": [16, 32],
        "output_heads": 5,
        "activation_layer": "ReLU",
        "norm_layer": None,
        "dropout": 0.1,
    }


@pytest.fixture
def sample_input():
    data = np.random.randn(64, 8)
    return AxisArray(
        data=data,
        dims=["time", "ch"],
        axes={
            "time": AxisArray.TimeAxis(fs=50.0),
            "ch": AxisArray.CoordinateAxis(data=np.arange(data.shape[1]), dims=["ch"]),
        },
        key="test_input",
    )


@pytest.fixture
def mlp_processor(mlp_settings):
    return TorchModelProcessor(
        model_class="ezmsg.learn.model.mlp.MLP",
        model_kwargs=mlp_settings,
        device="cpu",
    )


def test_mlp_forward_output_shape(sample_input, mlp_processor):
    result = mlp_processor(sample_input)[0]
    assert isinstance(result, AxisArray)
    assert result.data.shape[0] == sample_input.data.shape[0]
    assert result.data.shape[1] == 5
    assert "ch" in result.axes
    assert result.get_axis("ch").data.shape[0] == 5


def test_mlp_checkpoint_io(tmp_path, sample_input, mlp_settings):
    ckpt_file = tmp_path / "mlp_checkpoint.pth"

    proc1 = TorchModelProcessor(
        model_class="ezmsg.learn.model.mlp.MLP",
        model_kwargs=mlp_settings,
        device="cpu",
    )
    proc1(sample_input)
    proc1.save_checkpoint(str(ckpt_file))

    proc2 = TorchModelProcessor(
        model_class="ezmsg.learn.model.mlp.MLP",
        checkpoint_path=str(ckpt_file),
        model_kwargs=mlp_settings,
        device="cpu",
    )
    proc2(sample_input)

    state1 = proc1._state.model.state_dict()
    state2 = proc2._state.model.state_dict()

    for key in state1:
        assert torch.allclose(state1[key], state2[key], atol=1e-6)


def test_mlp_partial_fit_learns(sample_input, mlp_settings):
    from ezmsg.sigproc.sampler import SampleMessage, SampleTriggerMessage

    proc = TorchModelProcessor(
        model_class="ezmsg.learn.model.mlp.MLP",
        model_kwargs=mlp_settings,
        loss_fn=torch.nn.MSELoss(),
        learning_rate=0.01,
        device="cpu",
    )
    proc(sample_input)

    sample = AxisArray(
        data=sample_input.data[:1], dims=["time", "ch"], axes=sample_input.axes
    )
    target = np.random.randn(1, 5)

    msg = SampleMessage(
        sample=sample, trigger=SampleTriggerMessage(timestamp=0.0, value=target)
    )

    before = [p.detach().clone() for p in proc.state.model.parameters()]
    proc.partial_fit(msg)
    after = [p.detach().clone() for p in proc.state.model.parameters()]

    assert not all(torch.allclose(b, a) for b, a in zip(before, after))


@pytest.mark.parametrize("device", ["cpu", "cuda", "mps"])
def test_mlp_runs_on_available_devices(device, sample_input, mlp_settings):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    if device == "mps" and not torch.backends.mps.is_available():
        pytest.skip("MPS not available")

    proc = TorchModelProcessor(
        model_class="ezmsg.learn.model.mlp.MLP",
        model_kwargs=mlp_settings,
        device=device,
    )
    proc(sample_input)
    for p in proc._state.model.parameters():
        assert p.device.type == device


def test_mlp_hidden_size_integer(sample_input):
    proc = TorchModelProcessor(
        model_class="ezmsg.learn.model.mlp.MLP",
        model_kwargs={
            "input_size": 8,
            "hidden_size": 32,
            "num_layers": 3,
            "output_heads": 5,
            "activation_layer": "ReLU",
            "dropout": 0.1,
        },
        device="cpu",
    )
    proc(sample_input)
    hidden_layers = [
        m for m in proc._state.model.modules() if isinstance(m, torch.nn.Linear)
    ][:-1]  # Exclude the output head
    assert len(hidden_layers) == 3  # num_layers = 3
    assert hidden_layers[0].in_features == 8
    assert all(layer.out_features == 32 for layer in hidden_layers[:-1])
    assert hidden_layers[-1].out_features == 32


def test_mlp_rnn_style_missing_num_layers_raises(sample_input):
    with pytest.raises(ValueError, match="num_layers must be specified"):
        proc = TorchModelProcessor(
            model_class="ezmsg.learn.model.mlp.MLP",
            model_kwargs={
                "input_size": 8,
                "hidden_size": 64,
                "activation_layer": "torch.nn.ReLU",
            },
        )
        proc(sample_input)


def test_mlp_list_hidden_size_with_num_layers_mismatch(sample_input):
    with pytest.raises(ValueError, match="Length of hidden_size must match num_layers"):
        proc = TorchModelProcessor(
            model_class="ezmsg.learn.model.mlp.MLP",
            model_kwargs={
                "input_size": 8,
                "hidden_size": [32, 64, 10],
                "num_layers": 2,  # Mismatch: len(hidden_size) = 3
            },
        )
        proc(sample_input)


def test_mlp_empty_hidden_size_list(sample_input):
    with pytest.raises(ValueError, match="hidden_size must have at least one element"):
        proc = TorchModelProcessor(
            model_class="ezmsg.learn.model.mlp.MLP",
            model_kwargs={
                "input_size": 8,
                "hidden_size": [],
            },
        )
        proc(sample_input)


def test_mlp_multihead_output_keys(sample_input):
    proc = TorchModelProcessor(
        model_class="ezmsg.learn.model.mlp.MLP",
        model_kwargs={
            "input_size": 8,
            "hidden_size": [32],
            "output_heads": {"a": 3, "b": 2},
        },
        device="cpu",
    )
    outputs = proc(sample_input)
    keys = {o.key for o in outputs}
    assert keys == {"a", "b"}
    for o in outputs:
        assert o.data.shape[0] == sample_input.data.shape[0]
