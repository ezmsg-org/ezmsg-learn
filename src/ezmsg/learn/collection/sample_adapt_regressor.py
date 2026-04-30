from dataclasses import field

import ezmsg.core as ez
from ezmsg.baseproc import SampleTriggerMessage
from ezmsg.sigproc.resample import ResampleSettings, ResampleUnit
from ezmsg.sigproc.window import Window, WindowSettings
from ezmsg.util.messages.axisarray import AxisArray

from ezmsg.learn.process.adaptive_linear_regressor import (
    AdaptiveLinearRegressorSettings,
    AdaptiveLinearRegressorUnit,
)
from ezmsg.learn.process.flatten import Flatten, FlattenSettings
from ezmsg.learn.process.seqseqsampler import SeqSeqSamplerSettings, SeqSeqSamplerUnit
from ezmsg.learn.util import AdaptiveLinearRegressor


class SampleAdaptRegressorSettings(ez.Settings):
    # AdaptiveLinearRegressor settings
    model_type: AdaptiveLinearRegressor = AdaptiveLinearRegressor.LINEAR
    """Adaptive regressor backend/model."""

    model_path: str | None = None
    """Optional path to a pickled preconfigured model."""

    model_kwargs: dict = field(default_factory=dict)
    """Extra kwargs passed to the underlying regressor."""

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


class SampleAdaptRegressor(ez.Collection):
    SETTINGS = SampleAdaptRegressorSettings

    INPUT_LABELS = ez.InputTopic(AxisArray)
    INPUT_SIGNAL = ez.InputTopic(AxisArray)
    INPUT_TRIGGER = ez.InputTopic(SampleTriggerMessage)
    OUTPUT_SIGNAL = ez.OutputTopic(AxisArray)

    RESAMPLE = ResampleUnit()
    SEQSEQSAMPLER = SeqSeqSamplerUnit()
    WINDOW = Window()
    FLATTEN = Flatten()
    REGRESSOR = AdaptiveLinearRegressorUnit()

    def configure(self) -> None:
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
        self.WINDOW.apply_settings(
            WindowSettings(
                axis="time",
                newaxis="win",
                window_dur=self.SETTINGS.decode_window_dur,
                window_shift=self.SETTINGS.decode_window_shift,
                zero_pad_until="input",
            )
        )
        self.FLATTEN.apply_settings(
            FlattenSettings(
                preserve_axis="win",
                sample_axis="time",
                feature_axis="ch",
            )
        )
        self.REGRESSOR.apply_settings(
            AdaptiveLinearRegressorSettings(
                model_type=self.SETTINGS.model_type,
                settings_path=self.SETTINGS.model_path,
                model_kwargs=self.SETTINGS.model_kwargs,
            )
        )

    def network(self) -> ez.NetworkDefinition:
        network = [
            (self.INPUT_LABELS, self.RESAMPLE.INPUT_SIGNAL),
            (self.INPUT_SIGNAL, self.RESAMPLE.INPUT_REFERENCE),
            (self.RESAMPLE.OUTPUT_SIGNAL, self.SEQSEQSAMPLER.INPUT_VALUE),
            (self.INPUT_SIGNAL, self.SEQSEQSAMPLER.INPUT_SIGNAL),
            (self.INPUT_TRIGGER, self.SEQSEQSAMPLER.INPUT_TRIGGER),
            (self.SEQSEQSAMPLER.OUTPUT_SAMPLE, self.REGRESSOR.INPUT_SAMPLE),
        ]

        if self.SETTINGS.decode_window_dur is None:
            network.append((self.INPUT_SIGNAL, self.REGRESSOR.INPUT_SIGNAL))
        else:
            network.extend(
                [
                    (self.INPUT_SIGNAL, self.WINDOW.INPUT_SIGNAL),
                    (self.WINDOW.OUTPUT_SIGNAL, self.FLATTEN.INPUT_SIGNAL),
                    (self.FLATTEN.OUTPUT_SIGNAL, self.REGRESSOR.INPUT_SIGNAL),
                ]
            )

        network.append((self.REGRESSOR.OUTPUT_SIGNAL, self.OUTPUT_SIGNAL))
        return tuple(network)
