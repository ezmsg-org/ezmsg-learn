from pathlib import Path

import pytest
import numpy as np
from sklearn.model_selection import train_test_split
import torch
import torch.nn
from ezmsg.util.messages.axisarray import AxisArray
from ezmsg.sigproc.sampler import SampleMessage, SampleTriggerMessage

from ezmsg.learn.process.mlp_old import MLPProcessor


@pytest.mark.parametrize("norm_layer", [None, torch.nn.BatchNorm1d])
@pytest.mark.parametrize("activation_layer", [None, torch.nn.ReLU])
@pytest.mark.parametrize("inplace", [None, True, False])
@pytest.mark.parametrize("dropout", [0.0, 0.5])
def test_mlp_init(norm_layer, activation_layer, inplace, dropout):
    hidden_channels = [30, 20, 10]
    inplace = True
    bias = True
    single_precision = True

    proc = MLPProcessor(
        hidden_channels=hidden_channels,
        norm_layer=norm_layer,
        activation_layer=activation_layer,
        inplace=inplace,
        bias=bias,
        dropout=dropout,
        single_precision=single_precision,
    )

    # Verify the settings were registered properly
    assert proc.settings.hidden_channels == hidden_channels
    assert proc.settings.norm_layer == norm_layer
    assert proc.settings.activation_layer == activation_layer
    assert proc.settings.inplace == inplace
    assert proc.settings.bias == bias
    assert proc.settings.dropout == dropout
    assert proc.settings.single_precision == single_precision

    # The processor creates its model upon receipt of the first message, so let's create a message and send it.
    n_ch = 32
    data = np.arange(100 * n_ch).reshape(100, n_ch)
    ch_labels = np.array([f"ch{i}" for i in range(n_ch)])
    message = AxisArray(
        data=data,
        dims=["time", "ch"],
        axes={
            "time": AxisArray.TimeAxis(fs=512.0),
            "ch": AxisArray.CoordinateAxis(data=ch_labels, dims=["ch"]),
        },
        key="test_mlp_init",
    )
    proc(message)

    # Verify the settings were incorporated into the model
    mdl: torch.nn.Sequential = proc.state.model

    in_dim = n_ch
    idx = 0
    for layer_ix, nf in enumerate(hidden_channels):
        # Linear layer
        assert mdl[idx].in_features == in_dim
        assert mdl[idx].out_features == nf
        assert np.any(mdl[idx].bias.detach().numpy()) == bias

        # Optional normalization layer
        if norm_layer is not None and layer_ix < (len(hidden_channels) - 1):
            idx += 1
            assert isinstance(mdl[idx], norm_layer)
            assert mdl[idx].num_features == nf

        # Optional activation layer
        if activation_layer is not None and layer_ix < (len(hidden_channels) - 1):
            idx += 1
            assert isinstance(mdl[idx], activation_layer)

        # Dropout layer
        if layer_ix < (len(hidden_channels) - 1):
            idx += 1
            assert isinstance(mdl[idx], torch.nn.Dropout)
            assert mdl[idx].p == dropout

        in_dim = nf
        idx += 1


def test_mlp_process():
    # Generate a dataset with a non-linear relationship.
    n_ch = 32
    n_samps = 1_000_000
    data = np.random.randn(n_samps, n_ch)
    mat = np.random.randn(n_ch, 2)
    y = (data @ mat[:, 0]) ** 2 + (data @ mat[:, 1])

    # Normalize the data and targets
    data = (data - data.mean()) / data.std()
    y = (y - y.mean()) / y.std()

    # Segment the data into even-sampled segments
    batch_size = 100
    n_batches = n_samps // batch_size
    data = data.reshape(n_batches, -1, n_ch)
    y = y.reshape(n_batches, -1, 1)

    # Split into train and test sets
    test_size = 0.2
    X_train, X_test, y_train, y_test = train_test_split(
        data, y, test_size=test_size, shuffle=False
    )
    # Note: Split is NOT shuffled to simulate real-time data.
    assert np.array_equal(X_train, data[: X_train.shape[0]])
    half_test = int(test_size * n_batches // 2)

    # Create a template AxisArray that we will fill with data on-demand
    fs = 500.0
    template = AxisArray(
        data=np.zeros((batch_size, n_ch), dtype=float),
        dims=["time", "ch"],
        axes={
            "time": AxisArray.TimeAxis(fs=fs),
            "ch": AxisArray.CoordinateAxis(
                data=np.array([f"ch{i}" for i in range(n_ch)], dtype=str), dims=["ch"]
            ),
        },
        key="test_mlp_process",
    )

    def xy_gen(set: int = 0):
        if set == 0:
            X_batch, y_batch = X_train, y_train
        elif set == 1:
            X_batch, y_batch = X_test[:half_test], y_test[:half_test]
        else:
            X_batch, y_batch = X_test[half_test:], y_test[half_test:]
        for ix, (X, y) in enumerate(zip(X_batch, y_batch)):
            ts = (ix * batch_size) / fs
            # Normally we should use `replace` to create a new object and not mutate the incoming message.
            # axarr = replace(
            #     template,
            #     data=X,
            #     axes={
            #         **template.axes,
            #         "time": replace(template.axes["time"], offset=ts)
            #     },
            # )
            # But we are not using the incoming message for anything else, so we mutate the template for speed.
            template.data[:] = (
                X  # This would fail if n_samps / batch_size had a remainder.
            )
            template.axes["time"].offset = ts
            if set == 0:
                yield SampleMessage(
                    trigger=SampleTriggerMessage(timestamp=ts, value=y), sample=template
                )
            else:
                yield template, y

    # Create the processor
    proc = MLPProcessor(
        hidden_channels=[64, 1],
        norm_layer=None,  # torch.nn.LayerNorm,
        activation_layer=torch.nn.ReLU,
        inplace=False,
        bias=True,
        dropout=0.05,
        single_precision=True,
        learning_rate=1e-2,
        scheduler_gamma=1 - 1e-3,
    )

    # Run the adaptive training.
    result = []
    train_loss = []
    for sample_msg in xy_gen(set=0):
        # Naive closed-loop inference
        result.append(proc(sample_msg.sample))

        # Collect the loss to see if it decreases with training.
        train_loss.append(
            torch.nn.MSELoss()(
                torch.tensor(result[-1].data),
                torch.tensor(
                    sample_msg.trigger.value.reshape(-1, 1), dtype=torch.float32
                ),
            ).item()
        )

        # Train: This is unrealistic in that we would normally do inference on many axisarray messages throughout
        #  the trial, and only do training infrequently at the end of a trial if we can infer the labels.
        #  But I'm too lazy to split the data into many small axarrs and one large SampleMessage per trial.
        # Note: We don't have to call `partial_fit` because `__call__` inspects the message type and calls it for us.
        proc(sample_msg)

    def eval_test(processor, set: int = 1):
        # Run the test inference
        test_pred = []
        test_true = []
        for axarr, y in xy_gen(set=set):
            # Inference
            test_pred.append(processor(axarr).data)
            test_true.append(y)
        # Verify the model managed to learn the relationship
        test_pred = np.concatenate(test_pred).flatten()
        test_true = np.concatenate(test_true).flatten()
        ref = torch.tensor(np.mean(test_true) + np.zeros_like(test_true))
        test_loss = torch.nn.MSELoss()(
            torch.tensor(test_pred), torch.tensor(test_true)
        ).item()
        ref_loss = torch.nn.MSELoss()(ref, torch.tensor(test_true)).item()
        assert (test_loss / ref_loss) < 0.01
        """
        import matplotlib.pyplot as plt
        plt.subplot(2, 1, 1)
        plt.plot(np.log(np.array(train_loss)))
        plt.xlabel("Train Batch")
        plt.ylabel("Log Train Loss")
        plt.subplot(2, 1, 2)
        plt.plot(test_pred - test_true, label="diff")
        plt.xlabel("Test Sample")
        plt.ylabel("Test Residuals")
        plt.show()
        """

    eval_test(proc, set=1)

    # Save the checkpoint
    checkpoint_filename = "test_mlp_process.pth"
    proc.save_checkpoint(checkpoint_filename)

    # Create a new processor and load the checkpoint
    proc2 = MLPProcessor(
        hidden_channels=[64, 1],
        norm_layer=None,
        activation_layer=torch.nn.ReLU,
        inplace=False,
        bias=True,
        dropout=0.05,
        single_precision=True,
        learning_rate=1e-2,
        scheduler_gamma=1 - 1e-3,
        checkpoint_path=checkpoint_filename,
    )

    eval_test(proc2, set=2)

    Path(checkpoint_filename).unlink(missing_ok=True)
