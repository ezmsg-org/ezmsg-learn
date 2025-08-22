import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn
from ezmsg.sigproc.sampler import SampleMessage, SampleTriggerMessage
from ezmsg.util.messages.axisarray import AxisArray

from ezmsg.learn.process.transformer import TransformerProcessor


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
        key="test_transformer",
    )
    return message


@pytest.mark.parametrize("hidden_size,attention_heads", [(20, 4), (30, 5), (40, 4)])
@pytest.mark.parametrize("encoder_layers", [1, 2, 3])
@pytest.mark.parametrize("decoder_layers", [0, 1])
@pytest.mark.parametrize("output_size", [2, 3])
def test_transformer_init(
    hidden_size,
    attention_heads,
    encoder_layers,
    decoder_layers,
    output_size,
    simple_message,
):
    single_precision = True

    proc = TransformerProcessor(
        single_precision=single_precision,
        device="cpu",
        model_kwargs={
            "hidden_size": hidden_size,
            "encoder_layers": encoder_layers,
            "decoder_layers": decoder_layers,
            "output_size": output_size,
            "attention_heads": attention_heads,
        },
    )

    # Verify the settings were registered properly
    assert proc.settings.single_precision == single_precision

    # The processor creates its model upon receipt of the first message.
    proc(simple_message)

    # Verify the settings were incorporated into the model
    mdl: torch.nn.Module = proc.state.model

    in_dim = simple_message.data.shape[simple_message.get_axis_idx("ch")]

    assert mdl.input_proj.in_features == in_dim
    assert mdl.hidden_size == hidden_size
    assert len(mdl.encoder.layers) == encoder_layers
    for layer in mdl.encoder.layers:
        assert layer.dropout.p == 0.3
    assert mdl.heads["output"].out_features == output_size


@pytest.mark.parametrize("decoder_layers", [0, 1])
def test_transformer_process(simple_message, decoder_layers):
    hidden_size = 16
    encoder_layers = 1
    output_size = 2
    single_precision = True

    proc = TransformerProcessor(
        single_precision=single_precision,
        learning_rate=1e-2,
        scheduler_gamma=0.0,
        weight_decay=0.0,
        device="cpu",
        model_kwargs={
            "hidden_size": hidden_size,
            "encoder_layers": encoder_layers,
            "decoder_layers": decoder_layers,
            "output_size": output_size,
        },
    )

    output = proc(simple_message)[0]

    time_dim = (1,) if decoder_layers > 0 else simple_message.data.shape[:-1]
    assert output.data.shape == time_dim + (output_size,)

    # Bypass processor and call the model directly to verify output
    in_tensor = torch.tensor(simple_message.data[None, ...], dtype=torch.float32)
    with torch.no_grad():
        expected_result = proc.state.model(in_tensor)["output"].cpu().numpy().squeeze(0)
    assert np.allclose(output.data, expected_result)
    if decoder_layers > 0:
        assert proc.has_decoder
        # If decoder_layers > 0, tgt_cache should be initialized
        assert proc.state.tgt_cache is not None


@pytest.mark.parametrize("decoder_layers", [0, 1])
def test_transformer_partial_fit(simple_message, decoder_layers):
    hidden_size = 16
    encoder_layers = 1
    output_size = 2
    single_precision = True

    proc = TransformerProcessor(
        single_precision=single_precision,
        learning_rate=1e-2,
        scheduler_gamma=0.0,
        weight_decay=0.0,
        loss_fn=torch.nn.MSELoss(),
        device="cpu",
        model_kwargs={
            "hidden_size": hidden_size,
            "encoder_layers": encoder_layers,
            "decoder_layers": decoder_layers,
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

    proc.partial_fit(sample_message)

    assert not proc.state.model.training
    assert proc.state.tgt_cache is None
    updated_weights = [p.detach() for p in proc.state.model.parameters()]

    assert any(
        not torch.equal(w0, w1) for w0, w1 in zip(initial_weights, updated_weights)
    )


def test_transformer_checkpoint_save_load(simple_message):
    hidden_size = 16
    encoder_layers = 1
    output_size = 2
    single_precision = True

    proc = TransformerProcessor(
        single_precision=single_precision,
        device="cpu",
        model_kwargs={
            "hidden_size": hidden_size,
            "encoder_layers": encoder_layers,
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
        proc2 = TransformerProcessor(
            checkpoint_path=str(checkpoint_path),
            single_precision=single_precision,
            device="cpu",
            model_kwargs={
                "hidden_size": hidden_size,
                "encoder_layers": encoder_layers,
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


def test_transformer_partial_fit_multiloss(simple_message):
    hidden_size = 16
    encoder_layers = 1
    output_heads = {"traj": 3, "state": 3}
    single_precision = True

    proc = TransformerProcessor(
        single_precision=single_precision,
        learning_rate=1e-2,
        scheduler_gamma=0.0,
        weight_decay=0.0,
        loss_fn={"traj": torch.nn.MSELoss(), "state": torch.nn.CrossEntropyLoss()},
        loss_weights={"traj": 1.0, "state": 1.0},
        device="cpu",
        model_kwargs={
            "hidden_size": hidden_size,
            "encoder_layers": encoder_layers,
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
        np.random.randint(0, output_heads["state"], size=output[1].data.shape),
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


def test_autoregressive_cache_behavior(simple_message):
    proc = TransformerProcessor(
        single_precision=True,
        device="cpu",
        model_kwargs={
            "hidden_size": 8,
            "encoder_layers": 1,
            "decoder_layers": 1,  # Enable autoregressive mode
            "output_size": 2,
        },
    )

    # First call initializes model and cache
    proc(simple_message)
    cache1 = proc.state.tgt_cache.clone()

    # Second call should extend the cache
    proc(simple_message)
    cache2 = proc.state.tgt_cache

    assert cache2.shape[1] > cache1.shape[1]
    assert not torch.equal(cache1, cache2), "Cache should be updated with new data"


def test_cache_truncation(simple_message):
    max_len = 10

    proc = TransformerProcessor(
        single_precision=True,
        device="cpu",
        model_kwargs={
            "hidden_size": 8,
            "encoder_layers": 1,
            "decoder_layers": 1,
            "output_size": 2,
        },
        max_cache_len=max_len,
    )

    for _ in range(20):
        proc(simple_message)

    assert proc.state.tgt_cache.shape[1] <= max_len


def test_invalid_autoregressive_head_raises(simple_message):
    proc = TransformerProcessor(
        single_precision=True,
        device="cpu",
        loss_fn=torch.nn.MSELoss(),
        model_kwargs={
            "hidden_size": 8,
            "encoder_layers": 1,
            "decoder_layers": 1,
            "output_size": 2,
        },
        autoregressive_head="my_output_head",  # Invalid key
    )

    with pytest.raises(
        ValueError,
        match="Autoregressive head 'my_output_head' not found",
    ):
        proc(simple_message)
