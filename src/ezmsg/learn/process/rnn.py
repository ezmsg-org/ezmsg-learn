import typing

import ezmsg.core as ez
import numpy as np
import torch
from ezmsg.sigproc.base import BaseAdaptiveTransformer, BaseAdaptiveTransformerUnit
from ezmsg.sigproc.sampler import SampleMessage
from ezmsg.sigproc.util.profile import profile_subpub
from ezmsg.util.messages.axisarray import AxisArray
from ezmsg.util.messages.util import replace

from .base import ModelInitMixin
from .torch import (
    TorchModelSettings,
    TorchModelState,
    TorchProcessorMixin,
)


class RNNSettings(TorchModelSettings):
    model_class: str = "ezmsg.learn.model.rnn.RNNModel"
    """
    Fully qualified class path of the model to be used.
    This should be "ezmsg.learn.model.rnn.RNNModel" for this.
    """
    reset_hidden_on_fit: bool = True
    """
    Whether to reset the hidden state on each fit call.
    If True, the hidden state will be reset to zero after each fit.
    If False, the hidden state will be maintained across fit calls.
    """
    preserve_state_across_windows: bool | typing.Literal["auto"] = "auto"
    """
    Whether to preserve the hidden state across windows.
    If True, the hidden state will be preserved across windows.
    If False, the hidden state will be reset at the start of each window.
    If "auto", preserve if there is no overlap in time windows, otherwise reset.
    """


class RNNState(TorchModelState):
    hx: typing.Optional[torch.Tensor] = None


class RNNProcessor(
    BaseAdaptiveTransformer[RNNSettings, AxisArray, AxisArray, RNNState],
    TorchProcessorMixin,
    ModelInitMixin,
):
    def _infer_output_sizes(
        self, model: torch.nn.Module, n_input: int
    ) -> dict[str, int]:
        """Simple inference to get output channel size."""
        dummy_input = torch.zeros(1, 50, n_input, device=self._state.device)
        with torch.no_grad():
            output, _ = model(dummy_input)

        if isinstance(output, dict):
            return {k: v.shape[-1] for k, v in output.items()}
        else:
            return {"output": output.shape[-1]}

    def _reset_state(self, message: AxisArray) -> None:
        model_kwargs = dict(self.settings.model_kwargs or {})
        self._common_reset_state(message, model_kwargs)
        self._init_optimizer()
        self._validate_loss_keys(list(self._state.chan_ax.keys()))

        batch_size = 1 if message.data.ndim == 2 else message.data.shape[0]
        self.reset_hidden(batch_size)

    def _maybe_reset_state(self, message: AxisArray, batch_size: int) -> bool:
        preserve_state = self.settings.preserve_state_across_windows
        if preserve_state == "auto":
            axes = message.axes
            if batch_size < 2:
                # Single window, so preserve
                preserve_state = True
            elif "time" not in axes or "win" not in axes:
                # Default fallback
                ez.logger.warning(
                    "Missing 'time' or 'win' axis for auto preserve-state logic. Defaulting to reset."
                )
                preserve_state = False
            else:
                # Calculate stride between windows (assuming uniform spacing)
                win_stride = axes["win"].gain
                # Calculate window length from time axis length and gain
                time_len = message.data.shape[message.get_axis_idx("time")]
                gain = getattr(axes["time"], "gain", None)
                if gain is None:
                    ez.logger.warning(
                        "Time axis gain not found, using default gain of 1.0."
                    )
                    gain = 1.0  # fallback default
                win_len = time_len * gain
                # Determine if we should preserve state
                preserve_state = win_stride >= win_len

        # Preserve if windows do NOT overlap: stride >= window length
        if not preserve_state:
            self.reset_hidden(batch_size)
        else:
            # If preserving state, only reset if batch size isn't 1
            hx_batch_size = (
                self._state.hx[0].shape[1]
                if isinstance(self._state.hx, tuple)
                else self._state.hx.shape[1]
            )
            if hx_batch_size != 1:
                ez.logger.debug(
                    f"Resetting hidden state due to batch size mismatch (hx: {hx_batch_size}, new: 1)"
                )
                self.reset_hidden(1)
        return preserve_state

    def _process(self, message: AxisArray) -> list[AxisArray]:
        x = message.data
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(
                x,
                dtype=torch.float32
                if self.settings.single_precision
                else torch.float64,
                device=self._state.device,
            )

        # Add batch dimension if missing
        x, added_batch_dim = self._ensure_batched(x)

        batch_size = x.shape[0]
        preserve_state = self._maybe_reset_state(message, batch_size)

        with torch.no_grad():
            # If we are preserving state and have multiple batches, process sequentially
            if preserve_state and batch_size > 1:
                y_data = {}
                for x_batch in x:
                    x_batch = x_batch.unsqueeze(0)
                    y, self._state.hx = self._state.model(x_batch, hx=self._state.hx)
                    for key, out in y.items():
                        if key not in y_data:
                            y_data[key] = []
                        y_data[key].append(out.cpu().numpy())
                # Concatenate outputs for each key
                y_data = {
                    key: np.concatenate(outputs, axis=0)
                    for key, outputs in y_data.items()
                }
            else:
                y, self._state.hx = self._state.model(x, hx=self._state.hx)
                y_data = {
                    key: (
                        out.cpu().numpy().squeeze(0)
                        if added_batch_dim
                        else out.cpu().numpy()
                    )
                    for key, out in y.items()
                }

        return [
            replace(
                message,
                data=out,
                axes={**message.axes, "ch": self._state.chan_ax[key]},
                key=key,
            )
            for key, out in y_data.items()
        ]

    def reset_hidden(self, batch_size: int) -> None:
        self._state.hx = self._state.model.init_hidden(batch_size, self._state.device)

    def _train_step(
        self,
        X: torch.Tensor,
        y_targ: dict[str, torch.Tensor],
        loss_fns: dict[str, torch.nn.Module],
    ) -> None:
        y_pred, self._state.hx = self._state.model(X, hx=self._state.hx)
        if not isinstance(y_pred, dict):
            y_pred = {"output": y_pred}

        loss_weights = self.settings.loss_weights or {}
        losses = []
        for key in y_targ.keys():
            loss_fn = loss_fns.get(key)
            if loss_fn is None:
                raise ValueError(f"Loss function for key '{key}' is not defined.")
            if isinstance(loss_fn, torch.nn.CrossEntropyLoss):
                loss = loss_fn(y_pred[key].permute(0, 2, 1), y_targ[key].long())
            else:
                loss = loss_fn(y_pred[key], y_targ[key])
            weight = loss_weights.get(key, 1.0)
            losses.append(loss * weight)

        total_loss = sum(losses)
        ez.logger.debug(
            f"Training step loss: {total_loss.item()} (individual losses: {[loss.item() for loss in losses]})"
        )

        self._state.optimizer.zero_grad()
        total_loss.backward()
        self._state.optimizer.step()
        if self._state.scheduler is not None:
            self._state.scheduler.step()

    def partial_fit(self, message: SampleMessage) -> None:
        self._state.model.train()

        X = self._to_tensor(message.sample.data)

        # Add batch dimension if missing
        X, batched = self._ensure_batched(X)

        batch_size = X.shape[0]
        preserve_state = self._maybe_reset_state(message.sample, batch_size)

        y_targ = message.trigger.value
        if not isinstance(y_targ, dict):
            y_targ = {"output": y_targ}
        y_targ = {key: self._to_tensor(value) for key, value in y_targ.items()}
        # Add batch dimension to y_targ values if missing
        if batched:
            for key in y_targ:
                y_targ[key] = y_targ[key].unsqueeze(0)

        loss_fns = self.settings.loss_fn
        if loss_fns is None:
            raise ValueError("loss_fn must be provided in settings to use partial_fit")
        if not isinstance(loss_fns, dict):
            loss_fns = {k: loss_fns for k in y_targ.keys()}

        with torch.set_grad_enabled(True):
            if preserve_state:
                self._train_step(X, y_targ, loss_fns)
            else:
                for i in range(batch_size):
                    self._train_step(
                        X[i].unsqueeze(0),
                        {key: value[i].unsqueeze(0) for key, value in y_targ.items()},
                        loss_fns,
                    )

        self._state.model.eval()
        if self.settings.reset_hidden_on_fit:
            self.reset_hidden(X.shape[0])


class RNNUnit(
    BaseAdaptiveTransformerUnit[
        RNNSettings,
        AxisArray,
        AxisArray,
        RNNProcessor,
    ]
):
    SETTINGS = RNNSettings

    @ez.subscriber(BaseAdaptiveTransformerUnit.INPUT_SIGNAL, zero_copy=True)
    @ez.publisher(BaseAdaptiveTransformerUnit.OUTPUT_SIGNAL)
    @profile_subpub(trace_oldest=False)
    async def on_signal(self, message: AxisArray) -> typing.AsyncGenerator:
        results = await self.processor.__acall__(message)
        for result in results:
            yield self.OUTPUT_SIGNAL, result
