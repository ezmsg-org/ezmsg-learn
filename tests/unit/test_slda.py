import pytest
import numpy as np
import pickle
import tempfile
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from ezmsg.util.messages.axisarray import AxisArray
from ezmsg.learn.process.slda import SLDATransformer, SLDASettings


@pytest.fixture
def lda_model_file():
    # Create a simple LDA model for testing
    lda = LDA(solver="lsqr", shrinkage="auto")
    # Set up the model with a simple 2-class classification
    lda.classes_ = np.array([0, 1])
    lda.coef_ = np.array([[0.5, -0.5, 0.2]])
    lda.intercept_ = np.array([0.1])

    # Save the model to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pkl") as tmpfile:
        pickle.dump(lda, tmpfile)
        return tmpfile.name


@pytest.fixture
def slda_transformer(lda_model_file):
    settings = SLDASettings(settings_path=lda_model_file)
    return SLDATransformer(settings=settings)


@pytest.fixture
def sample_message():
    # Create a sample message with 3 channels and 2 time points
    data = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
    # Shape: (2, 3) - batch_size x time x channels
    return AxisArray(
        data=data,
        dims=["time", "ch"],
        axes={
            "time": AxisArray.TimeAxis(offset=0, fs=100.0),
            "ch": AxisArray.CoordinateAxis(data=np.array(["1", "2", "3"]), dims=["ch"]),
        },
        key="test",
    )


def test_init_and_reset_state(slda_transformer, sample_message):
    # Reset state should be called when the first message is processed
    slda_transformer._reset_state(sample_message)

    # Check if LDA model was loaded
    assert hasattr(slda_transformer.state, "lda")
    assert slda_transformer.state.lda.classes_.tolist() == [0, 1]

    # Check if output template was created
    assert slda_transformer.state.out_template is not None
    assert slda_transformer.state.out_template.dims == ["time", "classes"]
    assert slda_transformer.state.out_template.labels == [0, 1]
    assert slda_transformer.state.out_template.key == "test"


def test_process_sample(slda_transformer, sample_message):
    # Reset state first (normally done by __call__)
    slda_transformer._reset_state(sample_message)

    # Process the message
    result = slda_transformer._process(sample_message)

    # Check results
    assert result.data.shape == (2, 2)  # 2 time points, 2 classes
    assert result.dims == ["time", "classes"]
    assert result.labels == [0, 1]
    assert result.key == "test"

    # Probabilities should sum to 1 for each sample
    assert np.allclose(np.sum(result.data, axis=1), np.ones(2))


def test_empty_data(slda_transformer, sample_message):
    # Reset state first
    slda_transformer._reset_state(sample_message)

    # Create empty data message
    empty_data = np.zeros((0, 3))  # No time points
    empty_message = AxisArray(
        data=empty_data,
        dims=["time", "ch"],
        axes={
            "time": AxisArray.TimeAxis(offset=0, fs=100.0),
            "ch": AxisArray.CoordinateAxis(data=np.array(["1", "2", "3"]), dims=["ch"]),
        },
        key="empty",
    )

    # Process empty message
    result = slda_transformer._process(empty_message)

    # Should return empty template
    assert result.data.shape == (0, 2)  # 0 time points, 2 classes
    assert result.dims == ["time", "classes"]


def test_call(slda_transformer, sample_message):
    # Test __call__ method which calls _reset_state and _process
    result = slda_transformer(sample_message)

    # Verify result
    assert result.data.shape == (2, 2)
    assert result.dims == ["time", "classes"]
    np.testing.assert_allclose(np.sum(result.data, axis=1), np.ones(2))
