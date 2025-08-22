import importlib
import typing

import ezmsg.core as ez
import numpy as np
import torch
from ezmsg.sigproc.base import (
    BaseAdaptiveTransformer,
    BaseAdaptiveTransformerUnit,
    BaseStatefulTransformer,
    BaseTransformerUnit,
    processor_state,
)
from ezmsg.sigproc.sampler import SampleMessage
from ezmsg.sigproc.util.profile import profile_subpub
from ezmsg.util.messages.axisarray import AxisArray
from ezmsg.util.messages.util import replace

from .base import ModelInitMixin


class TorchSimpleSettings(ez.Settings):
    model_class: str
    """
    Fully qualified class path of the model to be used.
    Example: "my_module.MyModelClass"
    This class should inherit from `torch.nn.Module`.
    """

    checkpoint_path: str | None = None
    """
    Path to a checkpoint file containing model weights.
    If None, the model will be initialized with random weights.
    If parameters inferred from the weight sizes conflict with the settings / config,
    then the the inferred parameters will take priority and a warning will be logged.
    """

    config_path: str | None = None
    """
    Path to a config file containing model parameters.
    Parameters loaded from the config file will take priority over settings.
    If settings differ from config parameters then a warning will be logged.
    If `checkpoint_path` is provided then any parameters inferred from the weights
    will take priority over the config parameters.
    """

    single_precision: bool = True
    """Use single precision (float32) instead of double precision (float64)"""

    device: str | None = None
    """
    Device to use for the model. If None, the device will be determined automatically,
    with preference for cuda > mps > cpu.
    """

    model_kwargs: dict[str, typing.Any] | None = None
    """
    Additional keyword arguments to pass to the model constructor.
    This can include parameters like `input_size`, `output_size`, etc.
    If a config file is provided, these parameters will be updated with the config values.
    If a checkpoint file is provided, these parameters will be updated with the inferred values
    from the model weights.
    """


class TorchModelSettings(TorchSimpleSettings):
    learning_rate: float = 0.001
    """Learning rate for the optimizer"""

    weight_decay: float = 0.0001
    """Weight decay for the optimizer"""

    loss_fn: torch.nn.Module | dict[str, torch.nn.Module] | None = None
    """
    Loss function(s) for the decoder. If using multiple heads, this should be a dictionary
    mapping head names to loss functions. The keys must match the output head names.
    """

    loss_weights: dict[str, float] | None = None
    """
    Weights for each loss function if using multiple heads.
    The keys must match the output head names.
    If None or missing/mismatched keys, losses will be unweighted.
    """

    scheduler_gamma: float = 0.999
    """Learning scheduler decay rate. Set to 0.0 to disable the scheduler."""


@processor_state
class TorchSimpleState:
    model: torch.nn.Module | None = None
    device: torch.device | None = None
    chan_ax: dict[str, AxisArray.CoordinateAxis] | None = None


class TorchModelState(TorchSimpleState):
    optimizer: torch.optim.Optimizer | None = None
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None


P = typing.TypeVar("P", bound=BaseStatefulTransformer)


class TorchProcessorMixin:
    """Mixin with shared functionality for torch processors."""

    def _import_model(self, class_path: str) -> type[torch.nn.Module]:
        """Dynamically import model class from string."""
        if class_path is None:
            raise ValueError("Model class path must be provided in settings.")
        module_path, class_name = class_path.rsplit(".", 1)
        module = importlib.import_module(module_path)
        return getattr(module, class_name)

    def _infer_output_sizes(
        self: P, model: torch.nn.Module, n_input: int
    ) -> dict[str, int]:
        """Simple inference to get output channel size. Override if needed."""
        dummy_input = torch.zeros(1, 1, n_input, device=self._state.device)
        with torch.no_grad():
            output = model(dummy_input)

        if isinstance(output, dict):
            return {k: v.shape[-1] for k, v in output.items()}
        else:
            return {"output": output.shape[-1]}

    def _init_optimizer(self) -> None:
        self._state.optimizer = torch.optim.AdamW(
            self._state.model.parameters(),
            lr=self.settings.learning_rate,
            weight_decay=self.settings.weight_decay,
        )
        self._state.scheduler = (
            torch.optim.lr_scheduler.ExponentialLR(
                self._state.optimizer, gamma=self.settings.scheduler_gamma
            )
            if self.settings.scheduler_gamma > 0.0
            else None
        )

    def _validate_loss_keys(self, output_keys: list[str]):
        if isinstance(self.settings.loss_fn, dict):
            missing = [k for k in output_keys if k not in self.settings.loss_fn]
            if missing:
                raise ValueError(f"Missing loss function(s) for output keys: {missing}")

    def _to_tensor(self: P, data: np.ndarray) -> torch.Tensor:
        dtype = torch.float32 if self.settings.single_precision else torch.float64
        if isinstance(data, torch.Tensor):
            return data.detach().clone().to(device=self._state.device, dtype=dtype)
        return torch.tensor(data, dtype=dtype, device=self._state.device)

    def save_checkpoint(self: P, path: str) -> None:
        """Save the current model state to a checkpoint file."""
        if self._state.model is None:
            raise RuntimeError("Model must be initialized before saving a checkpoint.")

        checkpoint = {
            "model_state_dict": self._state.model.state_dict(),
            "config": self.settings.model_kwargs or {},
        }

        # Add optimizer state if available
        if hasattr(self._state, "optimizer") and self._state.optimizer is not None:
            checkpoint["optimizer_state_dict"] = self._state.optimizer.state_dict()

        torch.save(checkpoint, path)

    def _ensure_batched(self, tensor: torch.Tensor) -> tuple[torch.Tensor, bool]:
        """
        Ensure tensor has a batch dimension.
        Returns the potentially modified tensor and a flag indicating whether a dimension was added.
        """
        if tensor.ndim == 2:
            return tensor.unsqueeze(0), True
        return tensor, False

    def _common_process(self: P, message: AxisArray) -> list[AxisArray]:
        data = message.data
        data = self._to_tensor(data)

        # Add batch dimension if missing
        data, added_batch_dim = self._ensure_batched(data)

        with torch.no_grad():
            output = self._state.model(data)

        if isinstance(output, dict):
            output_messages = [
                replace(
                    message,
                    data=value.cpu().numpy().squeeze(0)
                    if added_batch_dim
                    else value.cpu().numpy(),
                    axes={
                        **message.axes,
                        "ch": self._state.chan_ax[key],
                    },
                    key=key,
                )
                for key, value in output.items()
            ]
            return output_messages

        return [
            replace(
                message,
                data=output.cpu().numpy().squeeze(0)
                if added_batch_dim
                else output.cpu().numpy(),
                axes={
                    **message.axes,
                    "ch": self._state.chan_ax["output"],
                },
            )
        ]

    def _common_reset_state(self: P, message: AxisArray, model_kwargs: dict) -> None:
        n_input = message.data.shape[message.get_axis_idx("ch")]

        if "input_size" in model_kwargs:
            if model_kwargs["input_size"] != n_input:
                raise ValueError(
                    f"Mismatch between model_kwargs['input_size']={model_kwargs['input_size']} "
                    f"and input data channels={n_input}"
                )
        else:
            model_kwargs["input_size"] = n_input

        device = (
            "cuda"
            if torch.cuda.is_available()
            else ("mps" if torch.mps.is_available() else "cpu")
        )
        device = self.settings.device or device
        self._state.device = torch.device(device)

        model_class = self._import_model(self.settings.model_class)

        self._state.model = self._init_model(
            model_class=model_class,
            params=model_kwargs,
            config_path=self.settings.config_path,
            checkpoint_path=self.settings.checkpoint_path,
            device=device,
        )

        self._state.model.eval()

        output_sizes = self._infer_output_sizes(self._state.model, n_input)
        self._state.chan_ax = {
            head: AxisArray.CoordinateAxis(
                data=np.array([f"{head}_ch{_}" for _ in range(size)]),
                dims=["ch"],
            )
            for head, size in output_sizes.items()
        }


class TorchSimpleProcessor(
    BaseStatefulTransformer[
        TorchSimpleSettings, AxisArray, AxisArray, TorchSimpleState
    ],
    TorchProcessorMixin,
    ModelInitMixin,
):
    def _reset_state(self, message: AxisArray) -> None:
        model_kwargs = dict(self.settings.model_kwargs or {})
        self._common_reset_state(message, model_kwargs)

    def _process(self, message: AxisArray) -> list[AxisArray]:
        """Process the input message and return the output messages."""
        return self._common_process(message)


class TorchSimpleUnit(
    BaseTransformerUnit[
        TorchSimpleSettings,
        AxisArray,
        AxisArray,
        TorchSimpleProcessor,
    ]
):
    SETTINGS = TorchSimpleSettings

    @ez.subscriber(BaseTransformerUnit.INPUT_SIGNAL, zero_copy=True)
    @ez.publisher(BaseTransformerUnit.OUTPUT_SIGNAL)
    @profile_subpub(trace_oldest=False)
    async def on_signal(self, message: AxisArray) -> typing.AsyncGenerator:
        results = await self.processor.__acall__(message)
        for result in results:
            yield self.OUTPUT_SIGNAL, result


class TorchModelProcessor(
    BaseAdaptiveTransformer[TorchModelSettings, AxisArray, AxisArray, TorchModelState],
    TorchProcessorMixin,
    ModelInitMixin,
):
    def _reset_state(self, message: AxisArray) -> None:
        model_kwargs = dict(self.settings.model_kwargs or {})
        self._common_reset_state(message, model_kwargs)
        self._init_optimizer()
        self._validate_loss_keys(list(self._state.chan_ax.keys()))

    def _process(self, message: AxisArray) -> list[AxisArray]:
        return self._common_process(message)

    def partial_fit(self, message: SampleMessage) -> None:
        self._state.model.train()

        X = self._to_tensor(message.sample.data)
        X, batched = self._ensure_batched(X)

        y_targ = message.trigger.value
        if not isinstance(y_targ, dict):
            y_targ = {"output": y_targ}
        y_targ = {key: self._to_tensor(value) for key, value in y_targ.items()}
        if batched:
            for key in y_targ:
                y_targ[key] = y_targ[key].unsqueeze(0)

        loss_fns = self.settings.loss_fn
        if loss_fns is None:
            raise ValueError("loss_fn must be provided in settings to use partial_fit")
        if not isinstance(loss_fns, dict):
            loss_fns = {k: loss_fns for k in y_targ.keys()}

        weights = self.settings.loss_weights or {}

        with torch.set_grad_enabled(True):
            y_pred = self._state.model(X)
            if not isinstance(y_pred, dict):
                y_pred = {"output": y_pred}

            losses = []
            for key in y_targ.keys():
                loss_fn = loss_fns.get(key)
                if loss_fn is None:
                    raise ValueError(
                        f"Loss function for key '{key}' is not defined in settings."
                    )
                if isinstance(loss_fn, torch.nn.CrossEntropyLoss):
                    loss = loss_fn(y_pred[key].permute(0, 2, 1), y_targ[key].long())
                else:
                    loss = loss_fn(y_pred[key], y_targ[key])
                weight = weights.get(key, 1.0)
                losses.append(loss * weight)
            total_loss = sum(losses)

            self._state.optimizer.zero_grad()
            total_loss.backward()
            self._state.optimizer.step()
            if self._state.scheduler is not None:
                self._state.scheduler.step()

        self._state.model.eval()


class TorchModelUnit(
    BaseAdaptiveTransformerUnit[
        TorchModelSettings,
        AxisArray,
        AxisArray,
        TorchModelProcessor,
    ]
):
    SETTINGS = TorchModelSettings

    @ez.subscriber(BaseAdaptiveTransformerUnit.INPUT_SIGNAL, zero_copy=True)
    @ez.publisher(BaseAdaptiveTransformerUnit.OUTPUT_SIGNAL)
    @profile_subpub(trace_oldest=False)
    async def on_signal(self, message: AxisArray) -> typing.AsyncGenerator:
        results = await self.processor.__acall__(message)
        for result in results:
            yield self.OUTPUT_SIGNAL, result
