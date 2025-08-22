import pickle
import tempfile
from pathlib import Path
import numpy as np
import pytest
from ezmsg.util.messages.axisarray import AxisArray

from ezmsg.learn.process.refit_kalman import (
    RefitKalmanFilterSettings,
    RefitKalmanFilterProcessor,
)


@pytest.fixture
def create_test_message():
    """Create a standard test message with synthetic neural data.

    Returns:
        AxisArray: Test message with 10 time steps, 2 channels, 100 Hz sampling rate
    """
    n_time, n_ch = 10, 2
    data = np.linspace(0, 1, n_time * n_ch).reshape(n_time, n_ch)
    fs = 100.0

    return AxisArray(
        data=data,
        dims=["time", "channels"],
        axes={
            "time": AxisArray.TimeAxis(fs=fs),
            "channels": AxisArray.CoordinateAxis(
                data=np.array([f"ch{i}" for i in range(n_ch)]),
                dims=["channels"],
            ),
        },
        key="test_message",
    )


@pytest.fixture
def checkpoint_file():
    """Create a realistic refit-ready Kalman checkpoint file.

    Creates temporary pickle file with Kalman filter parameters for testing.
    File should be cleaned up after each test to prevent accumulation.

    Returns:
        str: Path to temporary checkpoint file
    """
    dt = 0.1
    A = np.array([[1, dt], [0, 1]])
    H = np.array([[1, 0], [0, 1]])
    W = np.eye(2) * 0.05
    Q = np.eye(2) * 0.1
    checkpoint_data = {
        "A_state_transition_matrix": A,
        "W_process_noise_covariance": W,
        "H_observation_matrix": H,
        "Q_measurement_noise_covariance": Q,
    }
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pkl", mode="wb") as f:
        pickle.dump(checkpoint_data, f)
        return f.name


@pytest.mark.parametrize("steady_state", [True, False])
def test_processor_initialization_with_checkpoint(checkpoint_file, steady_state):
    """Test that processor initializes correctly with checkpoint file.

    Verifies processor can load pre-trained model from checkpoint and mark it as fitted.
    Tests both steady-state and non-steady-state modes.

    Args:
        checkpoint_file: Path to temporary checkpoint file
        steady_state: Whether to use steady-state Kalman filter mode
    """
    settings = RefitKalmanFilterSettings(
        checkpoint_path=checkpoint_file, steady_state=steady_state
    )
    processor = RefitKalmanFilterProcessor(settings)

    assert processor._state.model is None

    processor._reset_state()
    assert processor._state.model is not None
    assert processor._state.model.is_fitted is True

    Path(checkpoint_file).unlink()


@pytest.mark.parametrize("steady_state", [True, False])
def test_processor_initialization_without_checkpoint(steady_state):
    """Test that processor initializes correctly without checkpoint file.

    Verifies processor creates unfitted model that requires training before processing.
    Tests both steady-state and non-steady-state modes.

    Args:
        steady_state: Whether to use steady-state Kalman filter mode
    """
    settings = RefitKalmanFilterSettings(steady_state=steady_state)
    processor = RefitKalmanFilterProcessor(settings)

    assert processor._state.model is None

    processor._reset_state()
    assert processor._state.model is not None
    assert processor._state.model.is_fitted is False


@pytest.mark.parametrize("steady_state", [True, False])
def test_message_processing_with_checkpoint(
    create_test_message, checkpoint_file, steady_state
):
    """Test that messages can be processed with pre-loaded checkpoint.

    Verifies processor can make predictions using pre-trained model parameters.
    Tests both steady-state and non-steady-state processing modes.

    Args:
        create_test_message: Test message fixture with synthetic neural data
        checkpoint_file: Path to temporary checkpoint file
        steady_state: Whether to use steady-state Kalman filter mode
    """
    settings = RefitKalmanFilterSettings(
        checkpoint_path=checkpoint_file, steady_state=steady_state
    )
    processor = RefitKalmanFilterProcessor(settings)
    msg = create_test_message

    processor._reset_state(msg)
    result = processor._process(msg)

    assert processor._state.model is not None
    assert isinstance(result, AxisArray)
    assert result.data.shape == (
        10,
        processor._state.model.A_state_transition_matrix.shape[0],
    )
    assert np.all(np.isfinite(result.data))
    Path(checkpoint_file).unlink()


@pytest.mark.parametrize("steady_state", [True, False])
def test_message_processing_without_checkpoint_requires_fit(
    create_test_message, steady_state
):
    """Test that processing without checkpoint requires model fitting first.

    Verifies processor prevents processing when no fitted model is available.
    Should raise appropriate error to prevent invalid operations.

    Args:
        create_test_message: Test message fixture with synthetic neural data
        steady_state: Whether to use steady-state Kalman filter mode

    Raises:
        AttributeError or ValueError: Expected when processing without fitted model
    """
    settings = RefitKalmanFilterSettings(steady_state=steady_state)
    processor = RefitKalmanFilterProcessor(settings)
    msg = create_test_message

    # Should raise error when processing without fitted model
    with pytest.raises((AttributeError, ValueError)):
        processor._process(msg)


def test_state_update_during_processing(create_test_message, checkpoint_file):
    """Test that processor state is updated during message processing.

    Verifies Kalman filter state evolves properly as new data is processed.
    State should change from initial values after processing.

    Args:
        create_test_message: Test message fixture with synthetic neural data
        checkpoint_file: Path to temporary checkpoint file
    """
    settings = RefitKalmanFilterSettings(
        checkpoint_path=checkpoint_file, steady_state=False
    )
    processor = RefitKalmanFilterProcessor(settings)
    msg = create_test_message

    processor._reset_state(msg)
    x_prior = np.array([0, 0])

    processor._process(msg)

    x_post = processor._state.x
    assert not np.allclose(x_prior, x_post)
    # assert np.all(np.isfinite(x_post))
    Path(checkpoint_file).unlink()


def test_fit_method_functionality(create_test_message):
    """Test fit method functionality for training the model.

    Verifies processor can train model using provided neural and state data.
    After fitting, model should be marked as fitted and able to process new data.

    Args:
        create_test_message: Test message fixture with synthetic neural data
    """
    settings = RefitKalmanFilterSettings(steady_state=True)
    processor = RefitKalmanFilterProcessor(settings)

    # Test refit with minimal parameters
    X_neural = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
    Y_state = np.array([[1.0, 2.0], [1.1, 2.1], [1.2, 2.2]])

    # Test that fit method works with the processor
    processor.fit(X_neural, Y_state)

    # Check that model parameters were updated
    assert processor._state.model.is_fitted

    # Test that processing works after fit
    msg = create_test_message
    result = processor._process(msg)
    assert result.data.shape == (10, 2)
    assert np.all(np.isfinite(result.data))


def test_refit_functionality_with_buffered_data(create_test_message, checkpoint_file):
    """Test refit functionality using buffered data from online processing.

    Verifies processor can perform online refitting using collected buffer data.
    Model parameters should change after refit, and buffers should be cleared.

    Args:
        create_test_message: Test message fixture with synthetic neural data
        checkpoint_file: Path to temporary checkpoint file
    """
    settings = RefitKalmanFilterSettings(
        checkpoint_path=checkpoint_file, steady_state=True
    )
    processor = RefitKalmanFilterProcessor(settings)
    msg = create_test_message
    processor._reset_state(msg)

    # Get initial model parameters
    with open(checkpoint_file, "rb") as f:
        checkpoint_data = pickle.load(f)
    H_initial = checkpoint_data["H_observation_matrix"]
    Q_initial = checkpoint_data["Q_measurement_noise_covariance"]

    # Log data for refit
    target_pos = np.array([1.0, 1.0])
    for i in range(10):
        processor.log_for_refit(msg, target_pos, hold_flag=False)

    # Perform refit
    processor.refit_model()

    # Check that model parameters changed
    H_after_refit = processor._state.model.H_observation_matrix
    Q_after_refit = processor._state.model.Q_measurement_noise_covariance

    assert not np.allclose(H_initial, H_after_refit)
    assert not np.allclose(Q_initial, Q_after_refit)

    # Check that buffers are cleared after refit
    assert len(processor._state.buffer_neural) == 0
    assert len(processor._state.buffer_state) == 0

    Path(checkpoint_file).unlink()


def test_refit_functionality_without_buffered_data(
    create_test_message, checkpoint_file
):
    """Test refit functionality when no buffered data exists.

    Verifies processor handles refit requests gracefully with empty buffers.
    Should not raise errors and buffers should remain empty.

    Args:
        create_test_message: Test message fixture with synthetic neural data
        checkpoint_file: Path to temporary checkpoint file
    """
    settings = RefitKalmanFilterSettings(
        checkpoint_path=checkpoint_file, steady_state=True
    )
    processor = RefitKalmanFilterProcessor(settings)
    processor._reset_state()

    # Try to refit with empty buffer (should not raise error)
    processor.refit_model()

    # Check that buffers remain empty
    assert len(processor._state.buffer_neural) == 0
    assert len(processor._state.buffer_state) == 0

    Path(checkpoint_file).unlink()


def test_partial_fit_functionality(create_test_message, checkpoint_file):
    """Test partial fit functionality for incremental model updates.

    Verifies processor can perform partial fitting using individual sample messages.
    Model parameters should be updated incrementally after partial fit.

    Args:
        create_test_message: Test message fixture with synthetic neural data
        checkpoint_file: Path to temporary checkpoint file
    """
    settings = RefitKalmanFilterSettings(
        checkpoint_path=checkpoint_file, steady_state=True
    )
    processor = RefitKalmanFilterProcessor(settings)
    msg = create_test_message
    processor._reset_state(msg)

    with open(checkpoint_file, "rb") as f:
        checkpoint_data = pickle.load(f)
    H_initial = checkpoint_data["H_observation_matrix"]
    Q_initial = checkpoint_data["Q_measurement_noise_covariance"]

    # Create a mock SampleMessage with the expected structure
    class MockSampleMessage:
        def __init__(self, neural_data, trigger_value):
            self.sample = type("obj", (object,), {"data": neural_data})()
            self.trigger = type("obj", (object,), {"value": trigger_value})()

    # Create test data
    neural_data = np.array(
        [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]
    )  # 3 samples, 2 channels
    trigger_value = {
        "Y_state": np.array(
            [[1.0, 2.0], [1.1, 2.1], [1.2, 2.2]]
        ),  # 3 samples, 2 states
        "intention_velocity_indices": 0,
        "target_positions": np.array([[1.0, 1.0], [1.1, 1.1], [1.2, 1.2]]),
        "cursor_positions": np.array([[0.0, 0.0], [0.1, 0.1], [0.2, 0.2]]),
        "hold_flags": [False, False, False],
    }

    mock_message = MockSampleMessage(neural_data, trigger_value)
    processor.partial_fit(mock_message)

    assert not np.allclose(H_initial, processor._state.model.H_observation_matrix)
    assert not np.allclose(
        Q_initial, processor._state.model.Q_measurement_noise_covariance
    )

    Path(checkpoint_file).unlink()


def test_hold_periods_functionality(create_test_message, checkpoint_file):
    """Test hold periods functionality in refit data collection.

    Verifies processor correctly handles hold and non-hold periods during data collection.
    Buffers should be properly managed and cleared after refit.

    Args:
        create_test_message: Test message fixture with synthetic neural data
        checkpoint_file: Path to temporary checkpoint file
    """
    settings = RefitKalmanFilterSettings(
        checkpoint_path=checkpoint_file, steady_state=True
    )
    processor = RefitKalmanFilterProcessor(settings)
    msg = create_test_message
    processor._reset_state(msg)
    target_pos = np.array([1.0, 1.0])

    # Log data with mixed hold/non-hold periods
    for i in range(10):
        hold_flag = i % 2 == 0  # Alternate hold/non-hold
        processor.log_for_refit(msg, target_pos, hold_flag=hold_flag)

    # Perform refit
    processor.refit_model()

    # Check that buffers are cleared after refit
    assert len(processor._state.buffer_neural) == 0
    assert len(processor._state.buffer_state) == 0
    assert len(processor._state.buffer_cursor_positions) == 0
    assert len(processor._state.buffer_target_positions) == 0
    assert len(processor._state.buffer_hold_flags) == 0

    Path(checkpoint_file).unlink()


def test_message_processing_integration(create_test_message, checkpoint_file):
    """Test complete message processing integration workflow.

    Verifies complete integration from initialization through message processing.
    All components should work together to process neural data and produce valid output.

    Args:
        create_test_message: Test message fixture with synthetic neural data
        checkpoint_file: Path to temporary checkpoint file
    """
    settings = RefitKalmanFilterSettings(
        checkpoint_path=checkpoint_file, steady_state=True
    )
    processor = RefitKalmanFilterProcessor(settings)

    msg = create_test_message
    processor._reset_state(msg)
    result = processor._process(msg)

    # Check that processing worked
    assert isinstance(result, AxisArray)
    assert result.data.shape == (10, 2)
    assert np.all(np.isfinite(result.data))

    # Test that state was updated
    assert processor._state.x is not None
    assert np.all(np.isfinite(processor._state.x))

    Path(checkpoint_file).unlink()


def test_checkpoint_save_and_load_functionality(create_test_message, checkpoint_file):
    """Test checkpoint saving and loading functionality.

    Verifies processor can save model state and new processor can load it successfully.
    All model parameters should be preserved across save/load cycles.

    Args:
        create_test_message: Test message fixture with synthetic neural data
        checkpoint_file: Path to temporary checkpoint file
    """
    settings = RefitKalmanFilterSettings(checkpoint_path=checkpoint_file)
    processor = RefitKalmanFilterProcessor(settings)
    msg = create_test_message
    processor._reset_state(msg)
    # Save checkpoint
    new_checkpoint_path = checkpoint_file.replace(".pkl", "_new.pkl")
    processor.save_checkpoint(new_checkpoint_path)

    # Create new processor and load checkpoint
    settings_new = RefitKalmanFilterSettings(checkpoint_path=new_checkpoint_path)
    processor_new = RefitKalmanFilterProcessor(settings_new)
    processor_new._reset_state()
    # Check that models are equivalent
    assert np.allclose(
        processor._state.model.A_state_transition_matrix,
        processor_new._state.model.A_state_transition_matrix,
    )
    assert np.allclose(
        processor._state.model.H_observation_matrix,
        processor_new._state.model.H_observation_matrix,
    )
    assert np.allclose(
        processor._state.model.W_process_noise_covariance,
        processor_new._state.model.W_process_noise_covariance,
    )
    assert np.allclose(
        processor._state.model.Q_measurement_noise_covariance,
        processor_new._state.model.Q_measurement_noise_covariance,
    )

    Path(checkpoint_file).unlink()
    Path(new_checkpoint_path).unlink()


def test_error_handling_for_unfitted_model():
    """Test error handling for unfitted model during checkpoint saving.

    Verifies processor raises appropriate error when attempting to save checkpoint
    without fitted model. Should prevent creation of invalid checkpoint files.

    Raises:
        ValueError: Expected when attempting to save checkpoint without fitted model
    """
    # Test saving checkpoint without fitted model
    settings = RefitKalmanFilterSettings(checkpoint_path=None, steady_state=True)
    processor = RefitKalmanFilterProcessor(settings)

    with pytest.raises(ValueError, match="Cannot save checkpoint: model not fitted"):
        processor.save_checkpoint("test.pkl")


def test_error_handling_for_processing_without_checkpoint(create_test_message):
    """Test error handling for processing without checkpoint or fitted model.

    Verifies processor prevents processing when no fitted model is available.
    Should raise appropriate error to prevent invalid operations.

    Args:
        create_test_message: Test message fixture with synthetic neural data

    Raises:
        AttributeError or ValueError: Expected when processing without fitted model
    """
    settings = RefitKalmanFilterSettings(checkpoint_path=None, steady_state=False)
    processor = RefitKalmanFilterProcessor(settings)
    msg = create_test_message

    # Should raise error when processing without fitted model
    with pytest.raises((AttributeError, ValueError)):
        processor._process(msg)


def test_steady_state_vs_non_steady_state_processing(
    create_test_message, checkpoint_file
):
    """Test differences between steady-state and non-steady-state processing.

    Verifies processor produces different results in different modes due to
    different Kalman gain computation. Both modes should produce valid results.

    Args:
        create_test_message: Test message fixture with synthetic neural data
        checkpoint_file: Path to temporary checkpoint file
    """
    # Test steady-state mode
    settings_steady = RefitKalmanFilterSettings(
        checkpoint_path=checkpoint_file, steady_state=True
    )
    processor_steady = RefitKalmanFilterProcessor(settings_steady)
    msg = create_test_message
    processor_steady._reset_state(msg)

    result_steady = processor_steady._process(msg)

    # Test non-steady-state mode
    settings_nonsteady = RefitKalmanFilterSettings(
        checkpoint_path=checkpoint_file, steady_state=False
    )
    processor_nonsteady = RefitKalmanFilterProcessor(settings_nonsteady)
    processor_nonsteady._reset_state(msg)

    result_nonsteady = processor_nonsteady._process(msg)

    # Results should be different due to different Kalman gain computation
    assert not np.allclose(result_steady.data, result_nonsteady.data)
    assert np.all(np.isfinite(result_steady.data))
    assert np.all(np.isfinite(result_nonsteady.data))

    Path(checkpoint_file).unlink()
