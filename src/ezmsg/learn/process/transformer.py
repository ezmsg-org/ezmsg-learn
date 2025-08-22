import typing

import ezmsg.core as ez
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


class TransformerSettings(TorchModelSettings):
    model_class: str = "ezmsg.learn.model.transformer.TransformerModel"
    """
    Fully qualified class path of the model to be used.
    This should be "ezmsg.learn.model.transformer.TransformerModel" for this.
    """
    autoregressive_head: str | None = None
    """
    The name of the output head used for autoregressive decoding.
    This should match one of the keys in the model's output dictionary.
    If None, the first output head will be used.
    """
    max_cache_len: int | None = 128
    """
    Maximum length of the target sequence cache for autoregressive decoding.
    This limits the context length during decoding to prevent excessive memory usage.
    If set to None, the cache will grow indefinitely.
    """


class TransformerState(TorchModelState):
    ar_head: str | None = None
    """
    The name of the autoregressive head used for decoding.
    This is set based on the `autoregressive_head` setting.
    If None, the first output head will be used.
    """
    tgt_cache: typing.Optional[torch.Tensor] = None
    """
    Cache for the target sequence used in autoregressive decoding.
    This is updated with each processed message to maintain context.
    """


class TransformerProcessor(
    BaseAdaptiveTransformer[
        TransformerSettings, AxisArray, AxisArray, TransformerState
    ],
    TorchProcessorMixin,
    ModelInitMixin,
):
    @property
    def has_decoder(self) -> bool:
        return self.settings.model_kwargs.get("decoder_layers", 0) > 0

    def reset_cache(self) -> None:
        self._state.tgt_cache = None

    def _reset_state(self, message: AxisArray) -> None:
        model_kwargs = dict(self.settings.model_kwargs or {})
        self._common_reset_state(message, model_kwargs)
        self._init_optimizer()
        self._validate_loss_keys(list(self._state.chan_ax.keys()))

        self._state.tgt_cache = None
        if (
            self.settings.autoregressive_head is not None
            and self.settings.autoregressive_head not in self._state.chan_ax
        ):
            raise ValueError(
                f"Autoregressive head '{self.settings.autoregressive_head}' not found in target dictionary keys: {list(self._state.chan_ax.keys())}"
            )
        self._state.ar_head = (
            self.settings.autoregressive_head
            if self.settings.autoregressive_head is not None
            else list(self._state.chan_ax.keys())[0]
        )

    def _process(self, message: AxisArray) -> list[AxisArray]:
        # If has_decoder is False, fallback to regular processing
        if not self.has_decoder:
            return self._common_process(message)

        x = self._to_tensor(message.data)
        x, _ = self._ensure_batched(x)
        if x.shape[0] > 1:
            raise ValueError("Autoregressive decoding only supports batch size 1.")

        with torch.no_grad():
            y_pred = self._state.model(x, tgt=self._state.tgt_cache)

        pred = y_pred[self._state.ar_head]
        if self._state.tgt_cache is None:
            self._state.tgt_cache = pred[:, -1:, :]
        else:
            self._state.tgt_cache = torch.cat(
                [self._state.tgt_cache, pred[:, -1:, :]], dim=1
            )
        if self.settings.max_cache_len is not None:
            if self._state.tgt_cache.shape[1] > self.settings.max_cache_len:
                # Trim the cache to the maximum length
                self._state.tgt_cache = self._state.tgt_cache[
                    :, -self.settings.max_cache_len :, :
                ]

        if isinstance(y_pred, dict):
            return [
                replace(
                    message,
                    data=out.squeeze(0).cpu().numpy(),
                    axes={**message.axes, "ch": self._state.chan_ax[key]},
                    key=key,
                )
                for key, out in y_pred.items()
            ]
        else:
            return [
                replace(
                    message,
                    data=y_pred.squeeze(0).cpu().numpy(),
                    axes={**message.axes, "ch": self._state.chan_ax["output"]},
                )
            ]

    def partial_fit(self, message: SampleMessage) -> None:
        self._state.model.train()

        X = self._to_tensor(message.sample.data)
        X, batched = self._ensure_batched(X)

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

        weights = self.settings.loss_weights or {}

        if self.has_decoder:
            if X.shape[0] != 1:
                raise ValueError("Autoregressive decoding only supports batch size 1.")

            # Create shifted target for autoregressive head
            tgt_tensor = y_targ[self._state.ar_head]
            tgt = torch.cat(
                [
                    torch.zeros(
                        (1, 1, tgt_tensor.shape[-1]),
                        dtype=tgt_tensor.dtype,
                        device=tgt_tensor.device,
                    ),
                    tgt_tensor[:, :-1, :],
                ],
                dim=1,
            )

            # Reset tgt_cache at start of partial_fit to avoid stale context
            self.reset_cache()
            y_pred = self._state.model(X, tgt=tgt)
        else:
            # For non-autoregressive models, use the model directly
            y_pred = self._state.model(X)

        if not isinstance(y_pred, dict):
            y_pred = {"output": y_pred}

        with torch.set_grad_enabled(True):
            losses = []
            for key in y_targ.keys():
                loss_fn = loss_fns.get(key)
                if loss_fn is None:
                    raise ValueError(
                        f"Loss function for key '{key}' is not defined in settings."
                    )
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


class TransformerUnit(
    BaseAdaptiveTransformerUnit[
        TransformerSettings,
        AxisArray,
        AxisArray,
        TransformerProcessor,
    ]
):
    SETTINGS = TransformerSettings

    @ez.subscriber(BaseAdaptiveTransformerUnit.INPUT_SIGNAL, zero_copy=True)
    @ez.publisher(BaseAdaptiveTransformerUnit.OUTPUT_SIGNAL)
    @profile_subpub(trace_oldest=False)
    async def on_signal(self, message: AxisArray) -> typing.AsyncGenerator:
        results = await self.processor.__acall__(message)
        for result in results:
            yield self.OUTPUT_SIGNAL, result
