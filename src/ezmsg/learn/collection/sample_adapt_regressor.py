from dataclasses import field

import ezmsg.core as ez
from ezmsg.baseproc import SampleTriggerMessage
from ezmsg.sigproc.resample import ResampleSettings, ResampleUnit
from ezmsg.util.messages.axisarray import AxisArray

from ezmsg.learn.process.adaptive_linear_regressor import (
    AdaptiveLinearRegressorSettings,
    AdaptiveLinearRegressorUnit,
)
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


class SampleAdaptRegressor(ez.Collection):
    SETTINGS = SampleAdaptRegressorSettings

    INPUT_LABELS = ez.InputStream(AxisArray)
    INPUT_SIGNAL = ez.InputStream(AxisArray)
    INPUT_TRIGGER = ez.InputStream(SampleTriggerMessage)
    OUTPUT_SIGNAL = ez.OutputStream(AxisArray)

    RESAMPLE = ResampleUnit()
    SEQSEQSAMPLER = SeqSeqSamplerUnit()
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
        self.REGRESSOR.apply_settings(
            AdaptiveLinearRegressorSettings(
                model_type=self.SETTINGS.model_type,
                settings_path=self.SETTINGS.model_path,
                model_kwargs=self.SETTINGS.model_kwargs,
            )
        )

    def network(self) -> ez.NetworkDefinition:
        return (
            (self.INPUT_LABELS, self.RESAMPLE.INPUT_SIGNAL),
            (self.INPUT_SIGNAL, self.RESAMPLE.INPUT_REFERENCE),
            (self.RESAMPLE.OUTPUT_SIGNAL, self.SEQSEQSAMPLER.INPUT_VALUE),
            (self.INPUT_SIGNAL, self.SEQSEQSAMPLER.INPUT_SIGNAL),
            (self.INPUT_TRIGGER, self.SEQSEQSAMPLER.INPUT_TRIGGER),
            (self.SEQSEQSAMPLER.OUTPUT_SAMPLE, self.REGRESSOR.INPUT_SAMPLE),
            (self.INPUT_SIGNAL, self.REGRESSOR.INPUT_SIGNAL),
            (self.REGRESSOR.OUTPUT_SIGNAL, self.OUTPUT_SIGNAL),
        )
