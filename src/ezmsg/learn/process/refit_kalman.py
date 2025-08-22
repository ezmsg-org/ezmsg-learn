import pickle
from pathlib import Path

import ezmsg.core as ez
import numpy as np
from ezmsg.sigproc.base import (
    BaseAdaptiveTransformer,
    BaseAdaptiveTransformerUnit,
    processor_state,
)
from ezmsg.sigproc.sampler import SampleMessage
from ezmsg.util.messages.axisarray import AxisArray
from ezmsg.util.messages.util import replace

from ..model.refit_kalman import RefitKalmanFilter


class RefitKalmanFilterSettings(ez.Settings):
    """
    Settings for the Refit Kalman filter processor.

    This class defines the configuration parameters for the Refit Kalman filter processor.
    The RefitKalmanFilter is designed for online processing and playback.

    Attributes:
        checkpoint_path: Path to saved model parameters (optional).
            If provided, loads pre-trained parameters instead of learning from data.
        steady_state: Whether to use steady-state Kalman filter.
            If True, uses pre-computed Kalman gain; if False, updates dynamically.
    """

    checkpoint_path: str | None = None
    steady_state: bool = False
    velocity_indices: tuple[int, int] = (2, 3)


@processor_state
class RefitKalmanFilterState:
    """
    State management for the Refit Kalman filter processor.

    This class manages the persistent state of the Refit Kalman filter processor,
    including the model instance, current state estimates, and data buffers for refitting.

    Attributes:
        model: The RefitKalmanFilter model instance.
        x: Current state estimate (n_states,).
        P: Current state covariance matrix (n_states x n_states).
        buffer_neural: Buffer for storing neural activity data for refitting.
        buffer_state: Buffer for storing state estimates for refitting.
        buffer_cursor_positions: Buffer for storing cursor positions for refitting.
        buffer_target_positions: Buffer for storing target positions for refitting.
        buffer_hold_flags: Buffer for storing hold flags for refitting.
        current_position: Current cursor position estimate (2,).
    """

    model: RefitKalmanFilter | None = None
    x: np.ndarray | None = None
    P: np.ndarray | None = None

    buffer_neural: list | None = None
    buffer_state: list | None = None
    buffer_cursor_positions: list | None = None
    buffer_target_positions: list | None = None
    buffer_hold_flags: list | None = None


class RefitKalmanFilterProcessor(
    BaseAdaptiveTransformer[
        RefitKalmanFilterSettings,
        AxisArray,
        AxisArray,
        RefitKalmanFilterState,
    ]
):
    """
    Processor for implementing a Refit Kalman filter in the ezmsg framework.

    This processor integrates the RefitKalmanFilter model into the ezmsg
    message passing system. It handles the conversion between AxisArray messages
    and the internal Refit Kalman filter operations.

    The processor performs the following operations:
    1. Configures the Refit Kalman filter model with provided settings
    2. Processes incoming measurement messages
    3. Performs prediction and update steps
    4. Logs data for potential refitting
    5. Supports online refitting of the observation model
    6. Returns filtered state estimates as AxisArray messages
    7. Maintains state between message processing calls

    The processor can operate in two modes:
    1. Pre-trained mode: Loads parameters from checkpoint_path
    2. Learning mode: Collects data and fits the model when buffer is full

    Key features:
    - Online refitting capability for adaptive neural decoding
    - Data logging for retrospective analysis
    - Position tracking for cursor control applications
    - Hold period detection and handling

    Attributes:
        settings: Configuration settings for the Refit Kalman filter.
        _state: Internal state management object.

    Example:
        >>> # Create settings with checkpoint path
        >>> settings = RefitKalmanFilterSettings(
        ...     checkpoint_path="path/to/checkpoint.pkl",
        ...     steady_state=True
        ... )
        >>>
        >>> # Create processor
        >>> processor = RefitKalmanFilterProcessor(settings)
        >>>
        >>> # Process measurement message
        >>> result = processor(measurement_message)
        >>>
        >>> # Log data for refitting
        >>> processor.log_for_refit(message, target_pos, hold_flag)
        >>>
        >>> # Refit the model
        >>> processor.refit_model()
    """

    def _config_from_settings(self) -> dict:
        """
        Returns:
            dict: Dictionary containing configuration parameters for model initialization.
        """
        return {
            "steady_state": self.settings.steady_state,
        }

    def _init_model(self, **kwargs):
        """
        Initialize a new RefitKalmanFilter model with current settings.

        Args:
            **kwargs: Keyword arguments for model initialization.
        """
        config = self._config_from_settings()
        config.update(kwargs)
        self._state.model = RefitKalmanFilter(**config)

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        if self._state.model is None:
            self._init_model()
        if hasattr(self._state.model, "fit"):
            self._state.model.fit(X, y)

    def load_from_checkpoint(self, checkpoint_path: str) -> None:
        """
        Load model parameters from a serialized checkpoint file.

        Args:
            checkpoint_path (str): Path to the saved checkpoint file.

        Side Effects:
            - Initializes a new model if not already set.
            - Sets model matrices A, W, H, Q from the checkpoint.
            - Computes Kalman gain based on restored parameters.
        """
        checkpoint_file = Path(checkpoint_path)
        with open(checkpoint_file, "rb") as f:
            checkpoint_data = pickle.load(f)
        self._init_model(**checkpoint_data)
        self._state.model._compute_gain()
        self._state.model.is_fitted = True

    def save_checkpoint(self, checkpoint_path: str) -> None:
        """
        Save current model parameters to a checkpoint file.

        Args:
            checkpoint_path (str): Destination file path for saving model parameters.

        Raises:
            ValueError: If the model is not initialized or has not been fitted.
        """
        if not self._state.model or not self._state.model.is_fitted:
            raise ValueError("Cannot save checkpoint: model not fitted")
        checkpoint_data = {
            "A_state_transition_matrix": self._state.model.A_state_transition_matrix,
            "W_process_noise_covariance": self._state.model.W_process_noise_covariance,
            "H_observation_matrix": self._state.model.H_observation_matrix,
            "Q_measurement_noise_covariance": self._state.model.Q_measurement_noise_covariance,
        }
        checkpoint_file = Path(checkpoint_path)
        checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
        with open(checkpoint_file, "wb") as f:
            pickle.dump(checkpoint_data, f)

    def _reset_state(
        self,
        message: AxisArray = None,
    ):
        """
        This method initializes or reinitializes the state vector (x), state covariance (P),
        and cursor position. If a checkpoint path is specified in the settings, the model
        is loaded from the checkpoint.

        Args:
            message (AxisArray): Time-series message containing neural measurements.
            x_init (np.ndarray): Initial state vector.
            P_init (np.ndarray): Initial state covariance matrix.
        """
        if not self._state.model:
            if self.settings.checkpoint_path:
                self.load_from_checkpoint(self.settings.checkpoint_path)
            else:
                self._init_model()
                ## TODO: fit the model - how to do this given expected inputs X and y?
                # for unit test purposes only, given a known kinematic state size
                state_dim = 2

        # # If A is None, the model has not been fitted or loaded from checkpoint
        # if self._state.model.A_state_transition_matrix is None:
        #     raise RuntimeError(
        #         "Cannot reset state — model has not been fitted or loaded from checkpoint."
        #     )

        if self._state.model.A_state_transition_matrix is not None:
            state_dim = self._state.model.A_state_transition_matrix.shape[0]

        self._state.x = np.zeros(state_dim)
        self._state.P = np.eye(state_dim)

        self._state.buffer_neural = []
        self._state.buffer_state = []
        self._state.buffer_cursor_positions = []
        self._state.buffer_target_positions = []
        self._state.buffer_hold_flags = []

    def _process(self, message: AxisArray) -> AxisArray:
        """
        Process an incoming message using the Kalman filter.

        For each time point in the message:
            - Predict the next state
            - Update the estimate using the current measurement
            - Track the velocity and estimate position

        Args:
            message (AxisArray): Time-series message containing neural measurements.

        Returns:
            AxisArray: Filtered message containing updated state estimates.
        """
        # If checkpoint, load the model from the checkpoint
        if not self._state.model and self.settings.checkpoint_path:
            self.load_from_checkpoint(self.settings.checkpoint_path)
        # No checkpoint means you need to initialize and fit the model
        elif not self._state.model:
            self._init_model()
        state_dim = self._state.model.A_state_transition_matrix.shape[0]
        if self._state.x is None:
            self._state.x = np.zeros(state_dim)

        filtered_data = np.zeros(
            (
                message.data.shape[0],
                self._state.model.A_state_transition_matrix.shape[0],
            )
        )

        for i in range(message.data.shape[0]):
            measurement = message.data[i]
            # Predict
            x_pred, P_pred = self._state.model.predict(self._state.x)

            # Update
            x_updated = self._state.model.update(measurement, x_pred, P_pred)

            # Store
            self._state.x = x_updated.copy()
            self._state.P = self._state.model.P_state_covariance.copy()
            filtered_data[i] = self._state.x

        return replace(
            message,
            data=filtered_data,
            dims=["time", "state"],
            key=f"{message.key}_filtered" if hasattr(message, "key") else "filtered",
        )

    def partial_fit(self, message: SampleMessage) -> None:
        """
        Perform refitting using externally provided data.

        Expects message.sample.data (neural input) and message.trigger.value as a dict with:
            - Y_state: (n_samples, n_states) array
            - intention_velocity_indices: Optional[int]
            - target_positions: Optional[np.ndarray]
            - cursor_positions: Optional[np.ndarray]
            - hold_flags: Optional[list[bool]]
        """
        if not hasattr(message, "sample") or not hasattr(message, "trigger"):
            raise ValueError("Invalid message format for partial_fit.")

        X = np.array(message.sample.data)
        values = message.trigger.value

        if not isinstance(values, dict) or "Y_state" not in values:
            raise ValueError(
                "partial_fit expects trigger.value to include at least 'Y_state'."
            )

        kwargs = {
            "X_neural": X,
            "Y_state": np.array(values["Y_state"]),
        }

        # Optional fields
        for key in [
            "intention_velocity_indices",
            "target_positions",
            "cursor_positions",
            "hold_flags",
        ]:
            if key in values and values[key] is not None:
                kwargs[key if key != "hold_flags" else "hold_indices"] = np.array(
                    values[key]
                )

        # Call model refit
        self._state.model.refit(**kwargs)

    def log_for_refit(
        self,
        message: AxisArray,
        target_position: np.ndarray | None = None,
        hold_flag: bool | None = None,
    ):
        """
        Log data for potential refitting of the model.

        This method stores measurement data, state estimates, and contextual
        information (target positions, cursor positions, hold flags) in buffers
        for later use in refitting the observation model. This data is used
        to adapt the model to changing neural-to-behavioral relationships.

        Args:
            message: AxisArray message containing measurement data.
            target_position: Target position for the current time point (2,).
            hold_flag: Boolean flag indicating if this is a hold period.
        """
        if target_position is not None:
            self._state.buffer_target_positions.append(target_position.copy())
        if hold_flag is not None:
            self._state.buffer_hold_flags.append(hold_flag)

        measurement = message.data[-1]
        self._state.buffer_neural.append(measurement.copy())
        self._state.buffer_state.append(self._state.x.copy())

    def refit_model(self):
        """
        Refit the observation model (H, Q) using buffered measurements and contextual data.

        This method updates the model's understanding of the neural-to-state mapping
        by calculating a new observation matrix and noise covariance, based on:
            - Logged neural data
            - Cursor state estimates
            - Hold flags and target positions

        Args:
            velocity_indices (tuple): Indices in the state vector corresponding to velocity components.
                                    Default assumes 2D velocity at indices (0, 1).

        Raises:
            ValueError: If no buffered data exists.
        """
        if not self._state.buffer_neural:
            print("No buffered data to refit")
            return

        kwargs = {
            "X_neural": np.array(self._state.buffer_neural),
            "Y_state": np.array(self._state.buffer_state),
            "intention_velocity_indices": self.settings.velocity_indices[0],
        }

        if self._state.buffer_target_positions and self._state.buffer_cursor_positions:
            kwargs["target_positions"] = np.array(self._state.buffer_target_positions)
            kwargs["cursor_positions"] = np.array(self._state.buffer_cursor_positions)
        if self._state.buffer_hold_flags:
            kwargs["hold_indices"] = np.array(self._state.buffer_hold_flags)

        self._state.model.refit(**kwargs)

        self._state.buffer_neural.clear()
        self._state.buffer_state.clear()
        self._state.buffer_cursor_positions.clear()
        self._state.buffer_target_positions.clear()
        self._state.buffer_hold_flags.clear()


class RefitKalmanFilterUnit(
    BaseAdaptiveTransformerUnit[
        RefitKalmanFilterSettings,
        AxisArray,
        AxisArray,
        RefitKalmanFilterProcessor,
    ]
):
    SETTINGS = RefitKalmanFilterSettings
