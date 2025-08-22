import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn
from ezmsg.sigproc.sampler import SampleMessage, SampleTriggerMessage
from ezmsg.util.messages.axisarray import AxisArray

from ezmsg.learn.process.rnn import RNNProcessor


@pytest.fixture
def simple_message() -> AxisArray:
    n_ch = 192
    data = np.arange(100 * n_ch).reshape(100, n_ch)
    ch_labels = np.array([f"ch{i}" for i in range(n_ch)])
    message = AxisArray(
        data=data,
        dims=["time", "ch"],
        axes={
            "time": AxisArray.TimeAxis(fs=512.0),
            "ch": AxisArray.CoordinateAxis(data=ch_labels, dims=["ch"]),
        },
        key="test_rnn",
    )
    return message


@pytest.mark.parametrize("rnn_type", ["GRU", "LSTM", "RNN-Tanh", "RNN-ReLU"])
@pytest.mark.parametrize("hidden_size", [20, 30, 40])
@pytest.mark.parametrize("num_layers", [1, 2, 3])
@pytest.mark.parametrize("output_size", [2, 3])
def test_rnn_init(rnn_type, hidden_size, num_layers, output_size, simple_message):
    single_precision = True

    proc = RNNProcessor(
        single_precision=single_precision,
        device="cpu",
        model_kwargs={
            "hidden_size": hidden_size,
            "num_layers": num_layers,
            "output_size": output_size,
            "rnn_type": rnn_type,
        },
    )

    # Verify the settings were registered properly
    assert proc.settings.single_precision == single_precision

    # The processor creates its model upon receipt of the first message.
    proc(simple_message)

    # Verify the settings were incorporated into the model
    mdl: torch.nn.Module = proc.state.model

    in_dim = simple_message.data.shape[simple_message.get_axis_idx("ch")]

    assert mdl.linear_embeddings.in_features == in_dim
    assert mdl.rnn.input_size == in_dim
    assert mdl.rnn.hidden_size == hidden_size
    assert mdl.rnn.num_layers == num_layers
    assert mdl.rnn.dropout == 0.3 if num_layers > 1 else mdl.rnn.dropout == 0.0
    assert mdl.heads["output"].out_features == output_size

    rnn_mdl = list(mdl.children())[2]
    expected_module = {
        "GRU": torch.nn.GRU,
        "LSTM": torch.nn.LSTM,
        "RNN-Tanh": torch.nn.RNN,
        "RNN-ReLU": torch.nn.RNN,
    }[rnn_type]
    assert isinstance(rnn_mdl, expected_module)


@pytest.mark.parametrize("rnn_type", ["GRU", "LSTM", "RNN-Tanh", "RNN-ReLU"])
def test_rnn_process(rnn_type, simple_message):
    hidden_size = 16
    num_layers = 1
    output_size = 2
    single_precision = True

    proc = RNNProcessor(
        single_precision=single_precision,
        learning_rate=1e-2,
        scheduler_gamma=0.0,
        weight_decay=0.0,
        device="cpu",
        model_kwargs={
            "hidden_size": hidden_size,
            "num_layers": num_layers,
            "output_size": output_size,
            "rnn_type": rnn_type,
        },
    )

    output = proc(simple_message)[0]
    assert output.data.shape == simple_message.data.shape[:-1] + (output_size,)
    if rnn_type == "LSTM":
        for rnn_state in proc.state.hx:
            assert torch.any(rnn_state != 0)
    else:
        assert torch.any(proc.state.hx != 0)

    # Try calling the model directly and compare the result.
    # We don't pass in the hx state so it should be initialized to zeros, same as in the first call to proc.
    in_tensor = torch.tensor(simple_message.data[None, ...], dtype=torch.float32)
    with torch.no_grad():
        expected_result = (
            proc.state.model(in_tensor)[0]["output"].cpu().numpy().squeeze(0)
        )
    assert np.allclose(output.data, expected_result)


def test_rnn_partial_fit(simple_message):
    hidden_size = 16
    num_layers = 1
    output_size = 2
    single_precision = True

    proc = RNNProcessor(
        single_precision=single_precision,
        learning_rate=1e-2,
        scheduler_gamma=0.0,
        weight_decay=0.0,
        loss_fn=torch.nn.MSELoss(),
        device="cpu",
        model_kwargs={
            "hidden_size": hidden_size,
            "num_layers": num_layers,
            "output_size": output_size,
        },
    )

    proc(simple_message)  # Initialize the model

    initial_weights = [p.detach().clone() for p in proc.state.model.parameters()]

    target_shape = (simple_message.data.shape[0], output_size)
    target_value = np.ones(target_shape, dtype=np.float32)
    sample_message = SampleMessage(
        trigger=SampleTriggerMessage(timestamp=0.0, value=target_value),
        sample=simple_message,
    )

    proc(sample_message)

    assert not proc.state.model.training
    updated_weights = [p.detach() for p in proc.state.model.parameters()]

    assert any(
        not torch.equal(w0, w1) for w0, w1 in zip(initial_weights, updated_weights)
    )


def test_rnn_checkpoint_save_load(simple_message):
    hidden_size = 16
    num_layers = 1
    output_size = 2
    single_precision = True

    proc = RNNProcessor(
        single_precision=single_precision,
        device="cpu",
        model_kwargs={
            "hidden_size": hidden_size,
            "num_layers": num_layers,
            "output_size": output_size,
        },
    )

    # First pass to initialize model
    proc(simple_message)

    # Create a temporary file that is closed immediately
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
        checkpoint_path = Path(tmp.name)

    try:
        # Save full checkpoint (state_dict + config)
        proc.save_checkpoint(str(checkpoint_path))

        # Load from checkpoint
        proc2 = RNNProcessor(
            checkpoint_path=str(checkpoint_path),
            single_precision=single_precision,
            device="cpu",
            model_kwargs={
                "hidden_size": hidden_size,
                "num_layers": num_layers,
                "output_size": output_size,
            },
        )

        proc2(simple_message)

        # Compare state_dicts directly
        state_dict1 = proc.state.model.state_dict()
        state_dict2 = proc2.state.model.state_dict()

        for key in state_dict1:
            assert key in state_dict2, f"Missing key {key} in loaded state_dict"
            assert torch.equal(state_dict1[key], state_dict2[key]), (
                f"Mismatch in parameter {key}"
            )

    finally:
        # Ensure the temporary file is deleted
        checkpoint_path.unlink(missing_ok=True)


def test_rnn_partial_fit_multiloss(simple_message):
    hidden_size = 16
    num_layers = 1
    output_heads = {"traj": 3, "state": 3}
    single_precision = True

    proc = RNNProcessor(
        single_precision=single_precision,
        learning_rate=1e-2,
        scheduler_gamma=0.0,
        weight_decay=0.0,
        loss_fn={"traj": torch.nn.MSELoss(), "state": torch.nn.CrossEntropyLoss()},
        loss_weights={"traj": 1.0, "state": 1.0},
        device="cpu",
        model_kwargs={
            "hidden_size": hidden_size,
            "num_layers": num_layers,
            "output_size": output_heads,
        },
    )

    output = proc(simple_message)
    initial_weights = [p.detach().clone() for p in proc.state.model.parameters()]

    # Build targets
    traj_target = torch.tensor(
        np.random.randn(*output[0].data.shape),
        dtype=torch.float32,
    )
    state_target = torch.tensor(
        np.random.randint(0, output_heads["state"], size=output[1].data.shape[:-1]),
        dtype=torch.long,
    )

    sample_message = SampleMessage(
        trigger=SampleTriggerMessage(
            timestamp=0.0,
            value={"traj": traj_target, "state": state_target},
        ),
        sample=simple_message,
    )

    proc.partial_fit(sample_message)

    updated_weights = [p.detach() for p in proc.state.model.parameters()]
    assert any(
        not torch.equal(w0, w1) for w0, w1 in zip(initial_weights, updated_weights)
    )


@pytest.mark.parametrize(
    "preserve_state_across_windows, win_stride, win_len, should_preserve",
    [
        (True, 0.1, 0.1, True),
        (False, 0.1, 0.1, False),
        ("auto", 0.1, 0.1, True),  # non-overlapping → preserve
        ("auto", 0.05, 0.1, False),  # overlapping → reset
    ],
)
def test_rnn_preserve_state(
    preserve_state_across_windows, win_stride, win_len, should_preserve
):
    hidden_size = 16
    num_layers = 1
    output_size = 2
    single_precision = True

    proc = RNNProcessor(
        single_precision=single_precision,
        learning_rate=1e-2,
        scheduler_gamma=0.0,
        weight_decay=0.0,
        device="cpu",
        preserve_state_across_windows=preserve_state_across_windows,
        model_kwargs={
            "hidden_size": hidden_size,
            "num_layers": num_layers,
            "output_size": output_size,
        },
    )

    fs = 512.0
    n_time = int(fs * win_len)
    n_win = 3
    n_ch = 192
    ch_labels = np.array([f"ch{i}" for i in range(n_ch)])

    data = np.random.randn(n_win, n_time, n_ch).astype(np.float32)

    msg = AxisArray(
        data=data,
        dims=["win", "time", "ch"],
        axes={
            "time": AxisArray.TimeAxis(fs=fs),
            "win": AxisArray.LinearAxis(unit="s", gain=win_stride),
            "ch": AxisArray.CoordinateAxis(data=ch_labels, dims=["ch"]),
        },
        key="test_auto_param",
    )

    val0 = proc(msg)[0].data

    val1 = proc(msg)[0].data

    # Values should be the same when state is reset but different if state is preserved
    if should_preserve:
        assert not np.allclose(val0, val1)
    else:
        assert np.allclose(val0, val1)


def test_rnn_preserve_state_batch_size_change():
    hidden_size = 8
    output_size = 2
    n_ch = 192
    ch_labels = np.array([f"ch{i}" for i in range(n_ch)])

    proc = RNNProcessor(
        single_precision=True,
        device="cpu",
        preserve_state_across_windows=True,
        model_kwargs={"hidden_size": hidden_size, "output_size": output_size},
    )

    # First message: 1 window
    data1 = np.random.randn(1, 50, n_ch).astype(np.float32)
    msg1 = AxisArray(
        data=data1,
        dims=["win", "time", "ch"],
        axes={
            "time": AxisArray.TimeAxis(fs=512.0),
            "win": AxisArray.LinearAxis(unit="s", gain=0.1),
            "ch": AxisArray.CoordinateAxis(data=ch_labels, dims=["ch"]),
        },
        key="batch1",
    )

    # Second message: 3 windows
    data2 = np.random.randn(2, 50, n_ch).astype(np.float32)
    data2 = np.append(data2, data1, axis=0)
    msg2 = AxisArray(
        data=data2,
        dims=["win", "time", "ch"],
        axes={
            "time": AxisArray.TimeAxis(fs=512.0),
            "win": AxisArray.LinearAxis(unit="s", gain=0.1),
            "ch": AxisArray.CoordinateAxis(data=ch_labels, dims=["ch"]),
        },
        key="batch2",
    )

    val0 = proc(msg1)[0].data

    proc(msg2)
    val1 = proc(msg1)[0].data

    # Values should be different because state should not be reset
    assert not np.allclose(val0, val1)
