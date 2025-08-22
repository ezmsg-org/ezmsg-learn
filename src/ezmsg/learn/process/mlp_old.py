import typing

import numpy as np
import torch
import torch.nn
import ezmsg.core as ez
from ezmsg.util.messages.axisarray import AxisArray
from ezmsg.sigproc.sampler import SampleMessage
from ezmsg.util.messages.util import replace
from ezmsg.sigproc.base import (
    BaseAdaptiveTransformer,
    BaseAdaptiveTransformerUnit,
    processor_state,
)

from ..model.mlp_old import MLP


class MLPSettings(ez.Settings):
    hidden_channels: list[int]
    """List of the hidden channel dimensions"""

    norm_layer: typing.Callable[..., torch.nn.Module] | None = None
    """Norm layer that will be stacked on top of the linear layer. If None this layer won’t be used."""

    activation_layer: typing.Callable[..., torch.nn.Module] | None = torch.nn.ReLU
    """Activation function which will be stacked on top of the normalization layer (if not None), otherwise on top of the linear layer. If None this layer won’t be used."""

    inplace: bool | None = None
    """Parameter for the activation layer, which can optionally do the operation in-place. Default is None, which uses the respective default values of the activation_layer and Dropout layer."""

    bias: bool = True
    """Whether to use bias in the linear layer."""

    dropout: float = 0.0
    """The probability for the dropout layer."""

    single_precision: bool = True

    learning_rate: float = 0.001

    scheduler_gamma: float = 0.999
    """Learning scheduler decay rate. Set to 0.0 to disable the scheduler."""

    checkpoint_path: str | None = None
    """
    Path to a checkpoint file containing model weights.
    If None, the model will be initialized with random weights.
    """


@processor_state
class MLPState:
    model: MLP | None = None
    optimizer: torch.optim.Optimizer | None = None
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None
    template: AxisArray | None = None
    device: object | None = None


class MLPProcessor(
    BaseAdaptiveTransformer[MLPSettings, AxisArray, AxisArray, MLPState]
):
    def _hash_message(self, message: AxisArray) -> int:
        hash_items = (message.key,)
        if "ch" in message.dims:
            hash_items += (message.data.shape[message.get_axis_idx("ch")],)
        return hash(hash_items)

    def _reset_state(self, message: AxisArray) -> None:
        # Create the model
        self._state.model = MLP(
            in_channels=message.data.shape[message.get_axis_idx("ch")],
            hidden_channels=self.settings.hidden_channels,
            norm_layer=self.settings.norm_layer,
            activation_layer=self.settings.activation_layer,
            inplace=self.settings.inplace,
            bias=self.settings.bias,
            dropout=self.settings.dropout,
        )

        # Load model weights from checkpoint if specified
        if self.settings.checkpoint_path is not None:
            try:
                checkpoint = torch.load(self.settings.checkpoint_path)
                self._state.model.load_state_dict(checkpoint["model_state_dict"])
            except Exception as e:
                raise RuntimeError(
                    f"Failed to load checkpoint from {self.settings.checkpoint_path}: {str(e)}"
                )

        # Set the model to evaluation mode by default
        self._state.model.eval()

        # Create the optimizer
        self._state.optimizer = torch.optim.Adam(
            self._state.model.parameters(), lr=self.settings.learning_rate
        )

        # Update the optimizer from checkpoint if it exists
        if self.settings.checkpoint_path is not None:
            try:
                checkpoint = torch.load(self.settings.checkpoint_path)
                if "optimizer_state_dict" in checkpoint:
                    self._state.optimizer.load_state_dict(
                        checkpoint["optimizer_state_dict"]
                    )
            except Exception as e:
                raise RuntimeError(
                    f"Failed to load optimizer from {self.settings.checkpoint_path}: {str(e)}"
                )

        # TODO: Should the model be moved to a device before the next line?
        self._state.device = next(self.state.model.parameters()).device

        # Optionally create the learning rate scheduler
        self._state.scheduler = (
            torch.optim.lr_scheduler.ExponentialLR(
                self._state.optimizer, gamma=self.settings.scheduler_gamma
            )
            if self.settings.scheduler_gamma > 0.0
            else None
        )

        # Create the output channel axis for reuse in each output.
        n_output_channels = self.settings.hidden_channels[-1]
        self._state.chan_ax = AxisArray.CoordinateAxis(
            data=np.array([f"ch{_}" for _ in range(n_output_channels)]), dims=["ch"]
        )

    def save_checkpoint(self, path: str) -> None:
        """Save the current model state to a checkpoint file.

        Args:
            path: Path where the checkpoint will be saved
        """
        checkpoint = {
            "model_state_dict": self._state.model.state_dict(),
            "optimizer_state_dict": self._state.optimizer.state_dict(),
        }
        torch.save(checkpoint, path)

    def _to_tensor(self, data: np.ndarray) -> torch.Tensor:
        dtype = torch.float32 if self.settings.single_precision else torch.float64
        return torch.tensor(data, dtype=dtype, device=self._state.device)

    def partial_fit(self, message: SampleMessage) -> None:
        self._state.model.train()

        # TODO: loss_fn should be determined by setting
        loss_fn = torch.nn.functional.mse_loss

        X = self._to_tensor(message.sample.data)
        y_targ = self._to_tensor(message.trigger.value)

        with torch.set_grad_enabled(True):
            self._state.model.train()
            y_pred = self.state.model(X)
            loss = loss_fn(y_pred, y_targ)

            self.state.optimizer.zero_grad()
            loss.backward()
            self.state.optimizer.step()  # Update weights
            if self.state.scheduler is not None:
                self.state.scheduler.step()  # Update learning rate

        self._state.model.eval()

    def _process(self, message: AxisArray) -> AxisArray:
        data = message.data
        if not isinstance(data, torch.Tensor):
            data = torch.tensor(
                data,
                dtype=torch.float32
                if self.settings.single_precision
                else torch.float64,
            )

        with torch.no_grad():
            output = self.state.model(data.to(self.state.device))

        return replace(
            message,
            data=output.cpu().numpy(),
            axes={
                **message.axes,
                "ch": self.state.chan_ax,
            },
        )


class MLPUnit(
    BaseAdaptiveTransformerUnit[
        MLPSettings,
        AxisArray,
        AxisArray,
        MLPProcessor,
    ]
):
    SETTINGS = MLPSettings
