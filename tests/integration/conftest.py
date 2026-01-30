import ezmsg.core as ez
from ezmsg.baseproc import Clock, ClockSettings
from ezmsg.simbiophys.noise import WhiteNoise, WhiteNoiseSettings
from ezmsg.util.messages.axisarray import AxisArray


class NoiseSrcSettings(ez.Settings):
    fs: float = 10.0
    n_time: int = 4
    n_ch: int = 1
    dispatch_rate: float | None = None


class NoiseSrc(ez.Collection):
    """Self-contained multi-channel signal source for integration tests."""

    SETTINGS = NoiseSrcSettings

    OUTPUT_SIGNAL = ez.OutputStream(AxisArray)

    CLOCK = Clock()
    NOISE = WhiteNoise()

    def configure(self) -> None:
        dispatch_rate = self.SETTINGS.dispatch_rate or (self.SETTINGS.fs / self.SETTINGS.n_time)
        self.CLOCK.apply_settings(ClockSettings(dispatch_rate=dispatch_rate))
        self.NOISE.apply_settings(
            WhiteNoiseSettings(
                fs=self.SETTINGS.fs,
                n_time=self.SETTINGS.n_time,
                n_ch=self.SETTINGS.n_ch,
            )
        )

    def network(self) -> ez.NetworkDefinition:
        return (
            (self.CLOCK.OUTPUT_SIGNAL, self.NOISE.INPUT_CLOCK),
            (self.NOISE.OUTPUT_SIGNAL, self.OUTPUT_SIGNAL),
        )
