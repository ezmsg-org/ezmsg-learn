import pickle
import tempfile
import numpy as np
from pathlib import Path
import os

import ezmsg.core as ez
from ezmsg.util.messages.axisarray import AxisArray
from ezmsg.util.messagecodec import message_log
from ezmsg.util.messagelogger import MessageLogger, MessageLoggerSettings
from ezmsg.sigproc.synth import Counter, CounterSettings
from ezmsg.util.terminate import TerminateOnTotal, TerminateOnTimeout
from ezmsg.util.terminate import TerminateOnTotalSettings, TerminateOnTimeoutSettings

from ezmsg.learn.process.refit_kalman import (
    RefitKalmanFilterUnit,
    RefitKalmanFilterSettings,
)


class RefitKalmanSystemSettings(ez.Settings):
    counter_settings: CounterSettings
    unit_settings: RefitKalmanFilterSettings
    log_settings: MessageLoggerSettings
    terminate_total: TerminateOnTotalSettings
    terminate_timeout: TerminateOnTimeoutSettings


class RefitKalmanSystem(ez.Collection):
    SETTINGS = RefitKalmanSystemSettings

    SOURCE = Counter()
    UNIT = RefitKalmanFilterUnit()
    LOG = MessageLogger()
    TERM_TOTAL = TerminateOnTotal()
    TERM_TIMEOUT = TerminateOnTimeout()

    def configure(self) -> None:
        self.SOURCE.apply_settings(self.SETTINGS.counter_settings)
        self.UNIT.apply_settings(self.SETTINGS.unit_settings)
        self.LOG.apply_settings(self.SETTINGS.log_settings)
        self.TERM_TOTAL.apply_settings(self.SETTINGS.terminate_total)
        self.TERM_TIMEOUT.apply_settings(self.SETTINGS.terminate_timeout)

    def network(self) -> ez.NetworkDefinition:
        return (
            (self.SOURCE.OUTPUT_SIGNAL, self.UNIT.INPUT_SIGNAL),
            (self.UNIT.OUTPUT_SIGNAL, self.LOG.INPUT_MESSAGE),
            (self.LOG.OUTPUT_MESSAGE, self.TERM_TOTAL.INPUT_MESSAGE),
            (self.LOG.OUTPUT_MESSAGE, self.TERM_TIMEOUT.INPUT),
        )


def test_refit_kalman_system():
    """Test complete RefitKalmanFilter system integration.

    This test verifies that the RefitKalmanFilter can be successfully
    integrated into a complete ezmsg processing system. The test creates
    a realistic processing pipeline that generates synthetic neural signals,
    processes them through the Kalman filter, and logs the results.

    The integration test workflow:
    1. Create realistic Kalman filter checkpoint with model parameters
    2. Configure complete system with signal generation and processing
    3. Execute the system with specified duration and parameters
    4. Verify that messages are processed and logged correctly
    5. Validate output message format and data integrity

    Note:
        This test creates temporary files that are automatically cleaned up
        after execution. The test uses synthetic data and pre-trained model
        parameters to ensure consistent and reproducible results.
    """
    state_dim = 2
    duration = 2.0
    fs = 10.0
    block_size = 4
    test_path = Path(tempfile.gettempdir()) / "refit_kalman_log.txt"

    A = np.array([[1, 0.1], [0, 1]])
    H = np.array([[1, 0], [0, 1]])
    W = np.eye(2) * 0.05
    Q = np.eye(2) * 0.1

    # Write checkpoint
    checkpoint = {
        "A_state_transition_matrix": A,
        "H_observation_matrix": H,
        "W_process_noise_covariance": W,
        "Q_measurement_noise_covariance": Q,
    }
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pkl", mode="wb") as f:
        pickle.dump(checkpoint, f)
        checkpoint_file = f.name

    settings = RefitKalmanSystemSettings(
        counter_settings=CounterSettings(
            fs=fs,
            n_ch=1,
            n_time=block_size,
            dispatch_rate=duration,
        ),
        unit_settings=RefitKalmanFilterSettings(
            checkpoint_path=checkpoint_file,
            steady_state=True,
        ),
        log_settings=MessageLoggerSettings(output=test_path),
        terminate_total=TerminateOnTotalSettings(total=int(duration * fs / block_size)),
        terminate_timeout=TerminateOnTimeoutSettings(time=5.0),
    )

    system = RefitKalmanSystem(settings=settings)
    ez.run(SYSTEM=system)

    messages = [_ for _ in message_log(test_path)]
    os.remove(test_path)
    os.remove(checkpoint_file)

    assert len(messages) > 0, "No messages logged"

    for msg in messages:
        assert isinstance(msg, AxisArray)
        assert msg.dims == ["time", "state"]
        assert msg.data.ndim == 2
        assert msg.data.shape[1] == state_dim
        assert np.all(np.isfinite(msg.data))
