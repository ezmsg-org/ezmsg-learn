import ezmsg.core as ez
from ezmsg.baseproc import SampleTriggerMessage
from ezmsg.sigproc.resample import ResampleSettings, ResampleUnit
from ezmsg.util.messages.axisarray import AxisArray

from ezmsg.learn.process.adaptive_linear_regressor import (
    AdaptiveLinearRegressorSettings,
    AdaptiveLinearRegressorUnit,
)
from ezmsg.learn.process.seqseqsampler import SeqSeqSamplerSettings, SeqSeqSamplerUnit


class SampleAdaptRegressorSettings(ez.Settings):
    pass


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
        self.RESAMPLE.apply_settings(ResampleSettings())
        self.SEQSEQSAMPLER.apply_settings(SeqSeqSamplerSettings())
        self.REGRESSOR.apply_settings(AdaptiveLinearRegressorSettings())

    def network(self) -> ez.NetworkDefinition:
        return (
            (self.INPUT_LABELS, self.RESAMPLE.INPUT_SIGNAL),
            (self.INPUT_SIGNAL, self.RESAMPLE.INPUT_REFERENCE),
            (self.RESAMPLE.OUTPUT_SIGNAL, self.SEQSEQSAMPLER.INPUT_VALUE),
            (self.INPUT_SIGNAL, self.SEQSEQSAMPLER.INPUT_SIGNAL),
            (self.INPUT_TRIGGER, self.SEQSEQSAMPLER.INPUT_TRIGGER),
            (self.SEQSEQSAMPLER.OUTPUT_SAMPLE, self.REGRESSOR.INPUT_SAMPLE),
            (self.REGRESSOR.OUTPUT_SIGNAL, self.OUTPUT_SIGNAL),
        )
