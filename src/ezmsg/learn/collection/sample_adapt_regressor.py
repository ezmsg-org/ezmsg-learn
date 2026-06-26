from dataclasses import field

import ezmsg.core as ez
import numpy as np
from ezmsg.baseproc import (
    BaseStatefulTransformer,
    BaseTransformerUnit,
    SampleTriggerMessage,
    processor_state,
)
from ezmsg.sigproc.resample import ResampleSettings, ResampleUnit
from ezmsg.sigproc.window import Window, WindowSettings
from ezmsg.util.messages.axisarray import AxisArray
from ezmsg.util.messages.util import replace

from ezmsg.learn.process.adaptive_linear_regressor import (
    AdaptiveLinearRegressorSettings,
    AdaptiveLinearRegressorUnit,
)
from ezmsg.learn.process.flatten import Flatten, FlattenSettings
from ezmsg.learn.process.refit_kalman import (
    RefitKalmanFilterSettings,
    RefitKalmanFilterUnit,
)
from ezmsg.learn.process.seqseqsampler import SeqSeqSamplerSettings, SeqSeqSamplerUnit
from ezmsg.learn.process.torch import TorchModelSettings, TorchModelUnit
from ezmsg.learn.util import AdaptiveLinearRegressor

#: Default torch model class used when ``model_type == "mlp"``.
DEFAULT_TORCH_MODEL_CLASS = "ezmsg.learn.model.mlp.MLP"

#: ``model_type`` tokens routed to a non-linear regressor engine. Everything
#: else (``linear``/``logistic``/``sgd``/``par``/``ridge``) is handled by
#: :class:`AdaptiveLinearRegressorUnit` as before.
_TORCH_MODEL_TYPE = "mlp"
_KALMAN_MODEL_TYPE = "kalman"


def _model_type_token(model_type) -> str:
    if isinstance(model_type, AdaptiveLinearRegressor):
        return model_type.value
    return str(model_type).strip().lower()


def _model_backend(model_type) -> str:
    """Map ``model_type`` to the regressor engine that handles it:
    ``"torch"`` (MLP), ``"kalman"``, or ``"linear"`` (River/sklearn)."""
    token = _model_type_token(model_type)
    if token == _TORCH_MODEL_TYPE:
        return "torch"
    if token == _KALMAN_MODEL_TYPE:
        return "kalman"
    return "linear"


class DecodeOutputAdapterSettings(ez.Settings):
    output_labels: list | None = None
    """Channel labels for the decoded output. None -> generic ``ch0..chN``."""


@processor_state
class DecodeOutputAdapterState:
    ch_axis: AxisArray.CoordinateAxis | None = None


class DecodeOutputAdapterProcessor(
    BaseStatefulTransformer[
        DecodeOutputAdapterSettings,
        AxisArray,
        AxisArray,
        DecodeOutputAdapterState,
    ]
):
    """Normalize a decoder output into a ``(time, ch)`` AxisArray.

    The torch (``{"output": ...}``-keyed) and Kalman (``["time", "state"]``)
    engines emit differently-shaped outputs than the River/sklearn regressor.
    This rebuilds a uniform ``(time, ch=output_labels)`` message — keyed
    ``<input>_pred`` like :class:`AdaptiveLinearRegressorUnit` — so downstream
    consumers see one contract regardless of backend.
    """

    def _reset_state(self, message: AxisArray) -> None:
        if self.settings.output_labels is not None:
            self.state.ch_axis = AxisArray.CoordinateAxis(
                data=np.asarray(self.settings.output_labels), dims=["ch"]
            )

    def _process(self, message: AxisArray) -> AxisArray | None:
        data = np.asarray(message.data, dtype=float)
        if data.size == 0:
            return None

        if self.settings.output_labels is not None:
            n_outputs = len(self.settings.output_labels)
            data = data.reshape((-1, n_outputs))
            ch_axis = self.state.ch_axis
        else:
            data = data.reshape((data.shape[0], -1)) if data.ndim > 1 else data.reshape((1, -1))
            ch_axis = AxisArray.CoordinateAxis(
                data=np.asarray([f"ch{i}" for i in range(data.shape[-1])]), dims=["ch"]
            )

        # The decoder engines carry a ``time`` axis through (kalman keeps the
        # input's; the torch path inherits the windower's renamed ``win``->``time``
        # axis). Require it rather than silently emitting untimed samples — a
        # missing time axis means the upstream layout changed and downstream
        # timing/outlet behavior would be wrong.
        if "time" not in message.axes:
            raise ValueError(
                "DecodeOutputAdapter expected a 'time' axis on the decoder output "
                f"(got dims={message.dims}, axes={list(message.axes)}); the upstream "
                "windowing/regressor layout changed."
            )
        return replace(
            message,
            data=data,
            dims=["time", "ch"],
            axes={"ch": ch_axis, "time": message.axes["time"]},
            key=f"{message.key}_pred",
        )


class DecodeOutputAdapter(
    BaseTransformerUnit[
        DecodeOutputAdapterSettings,
        AxisArray,
        AxisArray,
        DecodeOutputAdapterProcessor,
    ]
):
    SETTINGS = DecodeOutputAdapterSettings


class SampleAdaptRegressorSettings(ez.Settings):
    # Regressor backend/model. Accepts the AdaptiveLinearRegressor enum (or its
    # string value) for the River/sklearn engines, plus the strings ``"mlp"``
    # and ``"kalman"`` which route to the torch / refit-Kalman engines.
    model_type: AdaptiveLinearRegressor | str = AdaptiveLinearRegressor.LINEAR
    """Regressor backend/model."""

    model_path: str | None = None
    """Optional path to a pre-trained checkpoint. Format depends on the
    backend: a pickled River/sklearn estimator, a ``torch.save`` artifact
    (mlp), or a pickled state-space matrix dict (kalman)."""

    model_kwargs: dict = field(default_factory=dict)
    """Extra kwargs passed to the underlying regressor."""

    # Torch (mlp) settings
    model_class: str = DEFAULT_TORCH_MODEL_CLASS
    """Fully-qualified torch model class used when ``model_type == "mlp"``."""

    device: str | None = None
    """Torch device for the mlp backend. None -> auto (cuda/mps/cpu)."""

    # Kalman settings
    steady_state: bool = True
    """Kalman steady-state gain flag, used when ``model_type == "kalman"``."""

    # Output adapter (mlp/kalman)
    output_labels: list | None = None
    """Decoded-output channel labels for the mlp/kalman adapter. None ->
    generic ``ch0..chN``."""

    # Resampling settings
    resample_axis: str = "time"
    """Axis to resample along."""

    resample_buffer_duration: float = 2.0
    """Duration of the buffer for resampling in seconds."""

    # SeqSeqSampler settings
    sampler_max_buffer_dur: float = 5.0
    """Maximum buffer duration for the SeqSeqSampler in seconds."""

    decode_window_dur: float | None = None
    """Optional inference-side feature window duration in seconds."""

    decode_window_shift: float | None = None
    """Optional inference-side feature window shift in seconds."""


def _build_regressor_unit(settings: SampleAdaptRegressorSettings):
    """Factory: construct the single regressor unit for ``settings.model_type``.

    Returns ``(unit, backend)`` where ``backend`` is ``"linear"`` (River/sklearn
    via :class:`AdaptiveLinearRegressorUnit`), ``"torch"`` (mlp), or ``"kalman"``.
    """
    backend = _model_backend(settings.model_type)
    if backend == "torch":
        return TorchModelUnit(), backend
    if backend == "kalman":
        return RefitKalmanFilterUnit(), backend
    return AdaptiveLinearRegressorUnit(), backend


def build_sample_adapt_regressor(
    settings: SampleAdaptRegressorSettings,
) -> ez.Collection:
    """Build a decode collection wired around a single regressor engine.

    The regressor backend (River/sklearn, torch-mlp, or refit-Kalman) is selected
    from ``settings.model_type`` and the collection class is defined dynamically
    so the graph contains exactly the units that backend uses — no inert,
    declared-but-unwired units. The signal path (and, for the linear engine, the
    online-adaptation sample path) wire to that one unit, so there is no per-
    backend wiring to keep in sync.
    """
    regressor, backend = _build_regressor_unit(settings)
    use_window = settings.decode_window_dur is not None
    use_sample_path = backend == "linear"  # online-adaptation path (River/sklearn)
    needs_adapter = backend != "linear"  # torch/kalman outputs need normalizing

    class SampleAdaptRegressor(ez.Collection):
        SETTINGS = SampleAdaptRegressorSettings

        INPUT_LABELS = ez.InputTopic(AxisArray)
        INPUT_SIGNAL = ez.InputTopic(AxisArray)
        INPUT_TRIGGER = ez.InputTopic(SampleTriggerMessage)
        OUTPUT_SIGNAL = ez.OutputTopic(AxisArray)

        REGRESSOR = regressor
        if use_window:
            WINDOW = Window()
            FLATTEN = Flatten()
        if use_sample_path:
            RESAMPLE = ResampleUnit()
            SEQSEQSAMPLER = SeqSeqSamplerUnit()
        if needs_adapter:
            ADAPTER = DecodeOutputAdapter()

        def configure(self) -> None:
            if backend == "linear":
                self.REGRESSOR.apply_settings(
                    AdaptiveLinearRegressorSettings(
                        model_type=self.SETTINGS.model_type,
                        settings_path=self.SETTINGS.model_path,
                        model_kwargs=self.SETTINGS.model_kwargs,
                    )
                )
            elif backend == "torch":
                self.REGRESSOR.apply_settings(
                    TorchModelSettings(
                        model_class=self.SETTINGS.model_class,
                        checkpoint_path=self.SETTINGS.model_path,
                        model_kwargs=dict(self.SETTINGS.model_kwargs),
                        device=self.SETTINGS.device,
                    )
                )
            else:
                self.REGRESSOR.apply_settings(
                    RefitKalmanFilterSettings(
                        checkpoint_path=self.SETTINGS.model_path,
                        steady_state=self.SETTINGS.steady_state,
                    )
                )

            if use_window:
                self.WINDOW.apply_settings(
                    WindowSettings(
                        axis="time",
                        newaxis="win",
                        window_dur=self.SETTINGS.decode_window_dur,
                        window_shift=self.SETTINGS.decode_window_shift,
                        # Window requires zero_pad_until="input" when
                        # window_shift is None (1:1 mode); "none" there only
                        # warns and is coerced to "input".
                        zero_pad_until="none"
                        if self.SETTINGS.decode_window_shift is not None
                        else "input",
                    )
                )
                self.FLATTEN.apply_settings(
                    FlattenSettings(
                        preserve_axis="win",
                        sample_axis="time",
                        feature_axis="ch",
                    )
                )
            if use_sample_path:
                self.RESAMPLE.apply_settings(
                    ResampleSettings(
                        axis=self.SETTINGS.resample_axis,
                        max_chunk_delay=float("inf"),
                        fill_value="extrapolate",
                        buffer_duration=self.SETTINGS.resample_buffer_duration,
                    )
                )
                self.SEQSEQSAMPLER.apply_settings(
                    SeqSeqSamplerSettings(
                        max_buffer_dur=self.SETTINGS.sampler_max_buffer_dur,
                    )
                )
            if needs_adapter:
                self.ADAPTER.apply_settings(
                    DecodeOutputAdapterSettings(
                        output_labels=self.SETTINGS.output_labels
                    )
                )

        def network(self) -> ez.NetworkDefinition:
            network = []
            if use_sample_path:
                # Online-adaptation sample path (River/sklearn only).
                network.extend(
                    [
                        (self.INPUT_LABELS, self.RESAMPLE.INPUT_SIGNAL),
                        (self.INPUT_SIGNAL, self.RESAMPLE.INPUT_REFERENCE),
                        (self.RESAMPLE.OUTPUT_SIGNAL, self.SEQSEQSAMPLER.INPUT_VALUE),
                        (self.INPUT_SIGNAL, self.SEQSEQSAMPLER.INPUT_SIGNAL),
                        (self.INPUT_TRIGGER, self.SEQSEQSAMPLER.INPUT_TRIGGER),
                        (self.SEQSEQSAMPLER.OUTPUT_SAMPLE, self.REGRESSOR.INPUT_SAMPLE),
                    ]
                )

            if use_window:
                network.extend(
                    [
                        (self.INPUT_SIGNAL, self.WINDOW.INPUT_SIGNAL),
                        (self.WINDOW.OUTPUT_SIGNAL, self.FLATTEN.INPUT_SIGNAL),
                        (self.FLATTEN.OUTPUT_SIGNAL, self.REGRESSOR.INPUT_SIGNAL),
                    ]
                )
            else:
                network.append((self.INPUT_SIGNAL, self.REGRESSOR.INPUT_SIGNAL))

            # River/sklearn already emits the canonical (time, ch) ``_pred``
            # contract; torch/kalman route through the adapter to match it.
            if needs_adapter:
                network.append((self.REGRESSOR.OUTPUT_SIGNAL, self.ADAPTER.INPUT_SIGNAL))
                network.append((self.ADAPTER.OUTPUT_SIGNAL, self.OUTPUT_SIGNAL))
            else:
                network.append((self.REGRESSOR.OUTPUT_SIGNAL, self.OUTPUT_SIGNAL))

            return tuple(network)

    return SampleAdaptRegressor(settings=settings)
