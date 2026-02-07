# refit_kalman.py
"""Refit Kalman filter for adaptive neural decoding.

.. note::
    This module supports the Array API standard via
    ``array_api_compat.get_namespace()``.  All linear algebra in :meth:`fit`,
    :meth:`predict`, and :meth:`update` stays in the source array namespace.
    The DARE solver in :meth:`_compute_gain` and the per-sample mutation loop
    in :meth:`refit` use NumPy regardless of input backend.
"""

import numpy as np
from array_api_compat import get_namespace
from ezmsg.sigproc.util.array import array_device, xp_asarray, xp_create
from numpy.linalg import LinAlgError
from scipy.linalg import solve_discrete_are


class RefitKalmanFilter:
    """
    Refit Kalman filter for adaptive neural decoding.

    This class implements a Kalman filter that can be refitted online during operation.
    Unlike the standard Kalman filter, this version can adapt its observation model
    (H and Q matrices) based on new data while maintaining the state transition model
    (A and W matrices). This is particularly useful for brain-computer interfaces
    where the relationship between neural activity and intended movements may change
    over time.

    The filter operates in two phases:
    1. Initial fitting: Learns all system matrices (A, W, H, Q) from training data
    2. Refitting: Updates only the observation model (H, Q) based on new data

    Attributes:
        A_state_transition_matrix: The state transition matrix A (n_states x n_states).
        W_process_noise_covariance: The process noise covariance matrix W (n_states x n_states).
        H_observation_matrix: The observation matrix H (n_observations x n_states).
        Q_measurement_noise_covariance: The measurement noise covariance matrix Q (n_observations x n_observations).
        K_kalman_gain: The Kalman gain matrix (n_states x n_observations).
        P_state_covariance: The state error covariance matrix (n_states x n_states).
        steady_state: Whether to use steady-state Kalman gain computation.
        is_fitted: Whether the model has been fitted with data.

    Example:
        >>> # Create and fit the filter
        >>> rkf = RefitKalmanFilter(steady_state=True)
        >>> rkf.fit(X_train, y_train)
        >>>
        >>> # Refit with new data
        >>> rkf.refit(X_new, Y_state, velocity_indices, targets, cursors, holds)
        >>>
        >>> # Predict with updated model
        >>> x_updated = rkf.predict_and_update(measurement, current_state)
    """

    def __init__(
        self,
        A_state_transition_matrix=None,
        W_process_noise_covariance=None,
        H_observation_matrix=None,
        Q_measurement_noise_covariance=None,
        steady_state=False,
        enforce_state_structure=False,
        alpha_fading_memory=1.000,
        process_noise_scale=1,
        measurement_noise_scale=1.2,
    ):
        self.A_state_transition_matrix = A_state_transition_matrix
        self.W_process_noise_covariance = W_process_noise_covariance
        self.H_observation_matrix = H_observation_matrix
        self.Q_measurement_noise_covariance = Q_measurement_noise_covariance
        self.K_kalman_gain = None
        self.P_state_covariance = None
        self.alpha_fading_memory = alpha_fading_memory

        # Noise scaling factors for smoothing control
        self.process_noise_scale = process_noise_scale
        self.measurement_noise_scale = measurement_noise_scale

        self.steady_state = steady_state
        self.enforce_state_structure = enforce_state_structure
        self.is_fitted = False

    def _validate_state_vector(self, Y_state):
        """
        Validate that the state vector has proper dimensions.

        Args:
            Y_state: State vector to validate

        Raises:
            ValueError: If state vector has invalid dimensions
        """
        if Y_state.ndim != 2:
            raise ValueError(f"State vector must be 2D, got {Y_state.ndim}D")

        if not hasattr(self, "H_observation_matrix") or self.H_observation_matrix is None:
            raise ValueError("Model must be fitted before refitting")

        expected_states = self.H_observation_matrix.shape[1]
        if Y_state.shape[1] != expected_states:
            raise ValueError(f"State vector has {Y_state.shape[1]} dimensions, expected {expected_states}")

    def fit(self, X_train, y_train):
        """
        Fit the Refit Kalman filter to the training data.

        This method learns all system matrices (A, W, H, Q) from training data
        using least-squares estimation, then computes the steady-state solution.
        This is the initial fitting phase that establishes the baseline model.

        Args:
            X_train: Neural activity (n_samples, n_neurons).
            y_train: Outputs being predicted (n_samples, n_states).

        Raises:
            ValueError: If training data has invalid dimensions.
            LinAlgError: If matrix operations fail during fitting.
        """
        # self._validate_state_vector(y_train)

        xp = get_namespace(X_train, y_train)
        _mT = xp.linalg.matrix_transpose

        X = xp.asarray(y_train)
        Z = xp.asarray(X_train)
        n_samples = X.shape[0]

        # Calculate the transition matrix (from x_t to x_t+1) using least-squares
        X2 = X[1:, :]  # x_{t+1}
        X1 = X[:-1, :]  # x_t
        A = _mT(X2) @ X1 @ xp.linalg.inv(_mT(X1) @ X1)  # Transition matrix
        W = _mT(X2 - X1 @ _mT(A)) @ (X2 - X1 @ _mT(A)) / (n_samples - 1)  # Covariance of transition matrix

        # Calculate the measurement matrix (from x_t to z_t) using least-squares
        H = _mT(Z) @ X @ xp.linalg.inv(_mT(X) @ X)  # Measurement matrix
        Q = _mT(Z - X @ _mT(H)) @ (Z - X @ _mT(H)) / Z.shape[0]  # Covariance of measurement matrix

        self.A_state_transition_matrix = A
        self.W_process_noise_covariance = W * self.process_noise_scale
        self.H_observation_matrix = H
        self.Q_measurement_noise_covariance = Q * self.measurement_noise_scale

        self._compute_gain()
        self.is_fitted = True

    def refit(
        self,
        X_neural,
        Y_state,
        intention_velocity_indices: int | None = None,
        target_positions=None,
        cursor_positions=None,
        hold_indices=None,
    ):
        """
        Refit the observation model based on new data.

        This method updates only the observation model (H and Q matrices) while
        keeping the state transition model (A and W matrices) unchanged. The refitting
        process modifies the intended states based on target positions and hold flags
        to better align with user intentions.

        The refitting process:
        1. Modifies intended states based on target positions and hold flags
        2. Recalculates the observation matrix H using least-squares
        3. Recalculates the measurement noise covariance Q
        4. Updates the Kalman gain accordingly

        Args:
            X_neural: Neural activity data (n_samples, n_neurons).
            Y_state: State estimates (n_samples, n_states).
            intention_velocity_indices: Index of velocity components in state vector.
            target_positions: Target positions for each sample (n_samples, 2).
            cursor_positions: Current cursor positions (n_samples, 2).
            hold_indices: Boolean flags indicating hold periods (n_samples,).

        Raises:
            ValueError: If input data has invalid dimensions or the model is not fitted.
        """
        self._validate_state_vector(Y_state)

        # Check if velocity indices are provided
        if intention_velocity_indices is None:
            # Assume (x, y, vx, vy)
            vel_idx = 2 if Y_state.shape[1] >= 4 else 0
            print(f"[RefitKalmanFilter] No velocity index provided — defaulting to {vel_idx}")
        else:
            if isinstance(intention_velocity_indices, (list, tuple)):
                if len(intention_velocity_indices) != 1:
                    raise ValueError("Only one velocity start index should be provided.")
                vel_idx = intention_velocity_indices[0]
            else:
                vel_idx = intention_velocity_indices

        # The per-sample mutation loop uses numpy for element-wise operations
        # on small vectors (np.linalg.norm on 2-element vectors, scalar indexing).
        Y_state_np = np.asarray(Y_state)
        target_positions_np = np.asarray(target_positions) if target_positions is not None else None
        cursor_positions_np = np.asarray(cursor_positions) if cursor_positions is not None else None

        # Only remap velocity if target and cursor positions are provided
        intended_states = Y_state_np.copy()
        if target_positions_np is not None and cursor_positions_np is not None:
            # Calculate intended velocities for each sample
            for i, (state, pos, target) in enumerate(zip(Y_state_np, cursor_positions_np, target_positions_np)):
                is_hold = hold_indices[i] if hold_indices is not None else False

                if is_hold:
                    # During hold periods, intended velocity is zero
                    intended_states[i, vel_idx : vel_idx + 2] = 0.0
                    if i > 0:
                        intended_states[i, :2] = intended_states[i - 1, :2]  # Same position as previous
                else:
                    # Calculate direction to target
                    to_target = target - pos
                    target_distance = np.linalg.norm(to_target)

                    if target_distance > 1e-5:  # Avoid division by zero
                        # Get current decoded velocity magnitude
                        current_velocity = state[vel_idx : vel_idx + 2]
                        current_speed = np.linalg.norm(current_velocity)

                        # Calculate intended velocity: same speed, but toward target
                        target_direction = to_target / target_distance
                        intended_velocity = target_direction * current_speed

                        # Update intended state with new velocity
                        intended_states[i, vel_idx : vel_idx + 2] = intended_velocity
                    # If target is very close, keep original velocity
                    else:
                        intended_states[i, vel_idx : vel_idx + 2] = state[vel_idx : vel_idx + 2]

        # Convert back to source namespace for final linalg
        xp = get_namespace(X_neural)
        dev = array_device(X_neural)
        _mT = xp.linalg.matrix_transpose

        intended_states = xp_asarray(xp, intended_states, device=dev)
        Z = xp.asarray(X_neural)

        # Recalculate observation matrix and noise covariance
        H = (
            _mT(Z) @ intended_states @ xp.linalg.pinv(_mT(intended_states) @ intended_states)
        )  # Using pinv() instead of inv() to avoid singular matrix errors
        Q = _mT(Z - intended_states @ _mT(H)) @ (Z - intended_states @ _mT(H)) / Z.shape[0]

        self.H_observation_matrix = H
        self.Q_measurement_noise_covariance = Q

        self._compute_gain()

    def _compute_gain(self):
        """
        Compute the Kalman gain matrix.

        This method computes the Kalman gain matrix based on the current system
        parameters. In steady-state mode, it solves the discrete-time algebraic
        Riccati equation to find the optimal steady-state gain. In non-steady-state
        mode, it computes the gain using the current covariance matrix.

        The DARE solver requires NumPy arrays; results are converted back to the
        source array namespace.

        Raises:
            LinAlgError: If the Riccati equation cannot be solved or matrix operations fail.
        """
        xp = get_namespace(self.A_state_transition_matrix)
        dev = array_device(self.A_state_transition_matrix)
        _mT = xp.linalg.matrix_transpose

        # Convert to numpy for DARE (no Array API equivalent)
        A_np = np.asarray(self.A_state_transition_matrix)
        H_np = np.asarray(self.H_observation_matrix)
        W_np = np.asarray(self.W_process_noise_covariance)
        Q_np = np.asarray(self.Q_measurement_noise_covariance)

        try:
            P_np = solve_discrete_are(A_np.T, H_np.T, W_np, Q_np)
            self.P_state_covariance = xp_asarray(xp, P_np, device=dev)
            S = (
                self.H_observation_matrix @ self.P_state_covariance @ _mT(self.H_observation_matrix)
                + self.Q_measurement_noise_covariance
            )
            self.K_kalman_gain = self.P_state_covariance @ _mT(self.H_observation_matrix) @ xp.linalg.inv(S)
        except LinAlgError:
            Q_reg_np = Q_np + 1e-7 * np.eye(Q_np.shape[0])

            try:
                P_np = solve_discrete_are(A_np.T, H_np.T, W_np, Q_reg_np)
                self.P_state_covariance = xp_asarray(xp, P_np, device=dev)
                Q_reg = xp_asarray(xp, Q_reg_np, device=dev)
                S = self.H_observation_matrix @ self.P_state_covariance @ _mT(self.H_observation_matrix) + Q_reg
                self.K_kalman_gain = self.P_state_covariance @ _mT(self.H_observation_matrix) @ xp.linalg.inv(S)
                print("Warning: Used regularized matrices for DARE solution")
            except LinAlgError:
                # Fallback to identity or manual initialization
                print("Warning: DARE failed, using identity covariance")
                self.P_state_covariance = xp_create(xp.eye, self.A_state_transition_matrix.shape[0], device=dev)
        # else:
        #     n_states = self.A_state_transition_matrix.shape[0]
        #     self.P_state_covariance = (
        #         np.eye(n_states) * 1000
        #     )  # Large initial uncertainty

        #     P_m = (
        #         self.A_state_transition_matrix
        #         @ self.P_state_covariance
        #         @ self.A_state_transition_matrix.T
        #         + self.W_process_noise_covariance
        #     )

        #     S = (
        #         self.H_observation_matrix @ P_m @ self.H_observation_matrix.T
        #         + self.Q_measurement_noise_covariance
        #     )

        #     self.K_kalman_gain = P_m @ self.H_observation_matrix.T @ np.linalg.pinv(S)

        #     I_mat = np.eye(self.A_state_transition_matrix.shape[0])
        #     self.P_state_covariance = (
        #         I_mat - self.K_kalman_gain @ self.H_observation_matrix
        #     ) @ P_m

    def predict(self, x_current):
        """
        Predict the next state and covariance.

        This method predicts the next state and covariance using the current state.
        """
        xp = get_namespace(x_current)
        _mT = xp.linalg.matrix_transpose

        x_predicted = self.A_state_transition_matrix @ x_current
        if self.steady_state is True:
            return x_predicted, None
        else:
            P_predicted = self.alpha_fading_memory**2 * (
                self.A_state_transition_matrix @ self.P_state_covariance @ _mT(self.A_state_transition_matrix)
                + self.W_process_noise_covariance
            )
            return x_predicted, P_predicted

    def update(
        self,
        z_measurement,
        x_predicted,
        P_predicted=None,
    ):
        """Update state estimate and covariance based on measurement z."""
        xp = get_namespace(z_measurement, x_predicted)
        dev = array_device(x_predicted)
        _mT = xp.linalg.matrix_transpose

        # Compute residual
        innovation = z_measurement - self.H_observation_matrix @ x_predicted

        if self.steady_state:
            x_updated = x_predicted + self.K_kalman_gain @ innovation
            return x_updated

        if P_predicted is None:
            raise ValueError("P_predicted must be provided for non-steady-state mode")

        # Non-steady-state mode
        # System uncertainty
        S = (
            self.H_observation_matrix @ P_predicted @ _mT(self.H_observation_matrix)
            + self.Q_measurement_noise_covariance
        )

        # Kalman gain
        K = P_predicted @ _mT(self.H_observation_matrix) @ xp.linalg.pinv(S)

        # Updated state
        x_updated = x_predicted + K @ innovation

        # Covariance update
        n = self.A_state_transition_matrix.shape[0]
        I_mat = xp_create(xp.eye, n, device=dev)
        P_updated = (I_mat - K @ self.H_observation_matrix) @ P_predicted @ _mT(
            I_mat - K @ self.H_observation_matrix
        ) + K @ self.Q_measurement_noise_covariance @ _mT(K)

        # Save updated values
        self.P_state_covariance = P_updated
        self.K_kalman_gain = K
        # self.S = S  # Optional: for diagnostics

        return x_updated
