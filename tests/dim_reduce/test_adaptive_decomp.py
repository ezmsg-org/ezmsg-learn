import pytest
import numpy as np
from numpy.testing import assert_array_almost_equal
from sklearn.decomposition import IncrementalPCA, MiniBatchNMF

from ezmsg.util.messages.axisarray import AxisArray
from ezmsg.learn.dim_reduce.adaptive_decomp import (
    IncrementalPCATransformer,
    IncrementalPCASettings,
    MiniBatchNMFTransformer,
    MiniBatchNMFSettings,
)


@pytest.fixture
def pca_test_data():
    # Create random data with predictable components
    np.random.seed(42)
    n_samples = 100
    n_features = 10
    n_components = 3

    # Create data with clear components
    components = np.random.randn(n_components, n_features)
    coefficients = np.random.randn(n_samples, n_components)
    data = np.dot(coefficients, components)

    # Add noise
    data += 0.1 * np.random.randn(n_samples, n_features)

    # Create AxisArray message
    message = AxisArray(
        data=data.reshape(n_samples, 1, n_features),
        dims=["time", "channel", "feature"],
        axes={
            "time": AxisArray.CoordinateAxis(
                data=np.arange(n_samples).astype(float), dims=["time"], unit="s"
            ),
            "channel": AxisArray.CoordinateAxis(
                data=np.array(["ch1"]), dims=["channel"], unit="channel"
            ),
            "feature": AxisArray.CoordinateAxis(
                data=np.arange(n_features).astype(str), dims=["feature"], unit="feature"
            ),
        },
        key="test",
    )

    return {
        "message": message,
        "data": data,
        "n_samples": n_samples,
        "n_features": n_features,
        "n_components": n_components,
    }


@pytest.fixture
def nmf_test_data():
    # Create non-negative random data
    np.random.seed(42)
    n_samples = 100
    n_features = 10
    n_components = 3

    # Create non-negative data with clear components
    components = np.abs(np.random.randn(n_components, n_features))
    coefficients = np.abs(np.random.randn(n_samples, n_components))
    data = np.dot(coefficients, components)

    # Create AxisArray message
    message = AxisArray(
        data=data.reshape(n_samples, 1, n_features),
        dims=["time", "channel", "feature"],
        axes={
            "time": AxisArray.CoordinateAxis(
                data=np.arange(n_samples).astype(float), dims=["time"], unit="s"
            ),
            "channel": AxisArray.CoordinateAxis(
                data=np.array(["ch1"]), dims=["channel"], unit="channel"
            ),
            "feature": AxisArray.CoordinateAxis(
                data=np.arange(n_features).astype(str), dims=["feature"], unit="feature"
            ),
        },
        key="test",
    )

    return {
        "message": message,
        "data": data,
        "n_samples": n_samples,
        "n_features": n_features,
        "n_components": n_components,
    }


class TestIncrementalPCATransformer:
    def test_initialization(self, pca_test_data):
        """Test that the transformer initializes correctly"""
        n_components = pca_test_data["n_components"]
        settings = IncrementalPCASettings(
            axis="feature", n_components=n_components, whiten=False
        )
        transformer = IncrementalPCATransformer(settings=settings)

        assert isinstance(transformer._state.estimator, IncrementalPCA)
        assert transformer._state.estimator.n_components == n_components
        assert transformer._state.estimator.whiten is False

    def test_partial_fit(self, pca_test_data):
        """Test partial fitting of the transformer"""
        n_components = pca_test_data["n_components"]
        n_features = pca_test_data["n_features"]
        message = pca_test_data["message"]

        settings = IncrementalPCASettings(axis="feature", n_components=n_components)
        transformer = IncrementalPCATransformer(settings=settings)

        # Partial fit
        transformer.partial_fit(message)

        # Check that the estimator has been fitted
        assert hasattr(transformer._state.estimator, "components_")
        assert transformer._state.estimator.components_.shape == (
            n_components,
            n_features,
        )

    def test_process_after_fit(self, pca_test_data):
        """Test that processing works after fitting"""
        n_samples = pca_test_data["n_samples"]
        n_components = pca_test_data["n_components"]
        message = pca_test_data["message"]
        data = pca_test_data["data"]

        settings = IncrementalPCASettings(axis="feature", n_components=n_components)
        transformer = IncrementalPCATransformer(settings=settings)

        # Partial fit
        transformer.partial_fit(message)

        # Process
        result = transformer(message)

        # Check output
        assert isinstance(result, AxisArray)
        assert result.data.shape == (n_samples, 1, n_components)
        assert result.dims == ["time", "channel", "feature"]

        # Verify transform is consistent with sklearn's transform
        expected = transformer._state.estimator.transform(data)
        expected = expected.reshape(n_samples, 1, n_components)
        assert_array_almost_equal(result.data, expected)

    def test_process_without_fit(self, pca_test_data):
        """Test that processing returns an empty template when not fitted"""
        n_components = pca_test_data["n_components"]
        message = pca_test_data["message"]

        settings = IncrementalPCASettings(axis="feature", n_components=n_components)
        transformer = IncrementalPCATransformer(settings=settings)

        # Process without fitting
        result = transformer(message)

        # Check that an empty template is returned
        assert isinstance(result, AxisArray)
        assert result.data.shape[0] == 0
        assert result.dims == ["time", "channel", "feature"]


class TestMiniBatchNMFTransformer:
    def test_initialization(self, nmf_test_data):
        """Test that the transformer initializes correctly"""
        n_components = nmf_test_data["n_components"]
        settings = MiniBatchNMFSettings(
            axis="feature", n_components=n_components, max_iter=100, tol=1e-4
        )
        transformer = MiniBatchNMFTransformer(settings=settings)

        assert isinstance(transformer._state.estimator, MiniBatchNMF)
        assert transformer._state.estimator.n_components == n_components
        assert transformer._state.estimator.max_iter == 100
        assert transformer._state.estimator.tol == 1e-4

    def test_partial_fit(self, nmf_test_data):
        """Test partial fitting of the transformer"""
        n_components = nmf_test_data["n_components"]
        n_features = nmf_test_data["n_features"]
        message = nmf_test_data["message"]

        settings = MiniBatchNMFSettings(
            axis="feature",
            n_components=n_components,
            max_iter=50,  # Reduce iterations for test speed
        )
        transformer = MiniBatchNMFTransformer(settings=settings)

        # Partial fit
        transformer.partial_fit(message)

        # Check that the estimator has been fitted
        assert hasattr(transformer._state.estimator, "components_")
        assert transformer._state.estimator.components_.shape == (
            n_components,
            n_features,
        )

    def test_process_after_fit(self, nmf_test_data):
        """Test that processing works after fitting"""
        n_samples = nmf_test_data["n_samples"]
        n_components = nmf_test_data["n_components"]
        message = nmf_test_data["message"]
        data = nmf_test_data["data"]

        settings = MiniBatchNMFSettings(
            axis="feature",
            n_components=n_components,
            max_iter=50,  # Reduce iterations for test speed
        )
        transformer = MiniBatchNMFTransformer(settings=settings)

        # Partial fit
        transformer.partial_fit(message)

        # Process
        result = transformer(message)

        # Check output
        assert isinstance(result, AxisArray)
        assert result.data.shape == (n_samples, 1, n_components)
        assert result.dims == ["time", "channel", "feature"]

        # Verify transform is consistent with sklearn's transform
        expected = transformer._state.estimator.transform(data)
        expected = expected.reshape(n_samples, 1, n_components)
        assert_array_almost_equal(result.data, expected)

    def test_process_without_fit(self, nmf_test_data):
        """Test that processing returns an empty template when not fitted"""
        n_components = nmf_test_data["n_components"]
        message = nmf_test_data["message"]

        settings = MiniBatchNMFSettings(axis="feature", n_components=n_components)
        transformer = MiniBatchNMFTransformer(settings=settings)

        # Process without fitting
        result = transformer(message)

        # Check that an empty template is returned
        assert isinstance(result, AxisArray)
        assert result.data.shape[0] == 0
        assert result.dims == ["time", "channel", "feature"]
