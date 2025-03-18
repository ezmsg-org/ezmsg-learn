import pytest
import numpy as np

from ezmsg.util.messages.axisarray import AxisArray
from ezmsg.learn.dim_reduce.incremental_decomp import (
    IncrementalDecompSettings,
    IncrementalDecompTransformer,
)
from ezmsg.learn.dim_reduce.adaptive_decomp import (
    IncrementalPCATransformer,
    MiniBatchNMFTransformer,
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
            "time": AxisArray.TimeAxis(fs=50.0, offset=0.0),
            "channel": AxisArray.CoordinateAxis(
                data=np.array(["ch1"]), dims=["channel"], unit="channel"
            ),
            "feature": AxisArray.CoordinateAxis(
                data=np.arange(n_features).astype(str), dims=["feature"], unit="feature"
            ),
        },
        key="test_incremental_decomp_pca",
    )

    return {
        "message": message,
        "data": data,
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
            "time": AxisArray.TimeAxis(fs=50.0, offset=0.0),
            "channel": AxisArray.CoordinateAxis(
                data=np.array(["ch1"]), dims=["channel"], unit="channel"
            ),
            "feature": AxisArray.CoordinateAxis(
                data=np.arange(n_features).astype(str), dims=["feature"], unit="feature"
            ),
        },
        key="test_incremental_decomp_nmf",
    )

    return {
        "message": message,
        "data": data,
        "n_components": n_components,
    }


class TestIncrementalDecompTransformer:
    @pytest.mark.parametrize("update_interval", [0.0, 0.1])
    def test_initialization_pca(self, pca_test_data, update_interval):
        """Test that the transformer initializes correctly with PCA method"""
        n_components = pca_test_data["n_components"]
        settings = IncrementalDecompSettings(
            axis="feature",
            n_components=n_components,
            method="pca",
            update_interval=update_interval,
            whiten=False,
        )
        transformer = IncrementalDecompTransformer(settings=settings)

        # Check that processors are initialized correctly
        assert "decomp" in transformer._procs
        assert isinstance(transformer._procs["decomp"], IncrementalPCATransformer)
        pca = transformer._procs["decomp"]
        assert pca.settings.axis == "feature"
        assert pca.settings.batch_size is None
        assert pca.settings.n_components == n_components
        assert pca.settings.whiten is False

        if update_interval > 0:
            assert "windowing" in transformer._procs
            win = transformer._procs["windowing"]
            assert win.settings.axis == "time"
            assert win.settings.window_dur == update_interval
            assert win.settings.window_shift == update_interval
            assert win.settings.zero_pad_until == "none"
        else:
            assert "windowing" not in transformer._procs

    @pytest.mark.parametrize("update_interval", [0.0, 0.1])
    def test_initialization_nmf(self, nmf_test_data, update_interval):
        """Test that the transformer initializes correctly with NMF method"""
        n_components = nmf_test_data["n_components"]
        settings = IncrementalDecompSettings(
            axis="feature",
            n_components=n_components,
            method="nmf",
            update_interval=update_interval,
        )
        transformer = IncrementalDecompTransformer(settings=settings)

        # Check that processors are initialized correctly
        assert "decomp" in transformer._procs
        assert isinstance(transformer._procs["decomp"], MiniBatchNMFTransformer)
        if update_interval > 0:
            assert "windowing" in transformer._procs
        else:
            assert (
                "windowing" not in transformer._procs
            )  # No windowing as update_interval=0

    @pytest.mark.parametrize(
        "method, test_data_fixture",
        [("pca", "pca_test_data"), ("nmf", "nmf_test_data")],
    )
    def test_process(self, method, test_data_fixture, request):
        """Test processing with different decomposition methods"""
        test_data = request.getfixturevalue(test_data_fixture)
        n_components = test_data["n_components"]
        message = test_data["message"]
        n_samples = message.data.shape[0]

        settings_kwargs = {
            "axis": "feature",
            "n_components": n_components,
            "method": method,
            "update_interval": 0.0,
        }

        if method == "nmf":
            settings_kwargs.update({"init": "random", "beta_loss": "frobenius"})
        elif method == "pca":
            settings_kwargs["whiten"] = False

        transformer = IncrementalDecompTransformer(
            settings=IncrementalDecompSettings(**settings_kwargs)
        )

        # First do partial_fit manually to ensure the model is trained
        transformer._procs["decomp"].partial_fit(message)

        # Then process
        result = transformer(message)

        # Check output
        assert isinstance(result, AxisArray)
        assert result.data.shape == (n_samples, 1, n_components)
        assert result.dims == ["time", "channel", "feature"]

    @pytest.mark.parametrize("update_interval", [0.25, 0.5, 0.9, 1.0])
    def test_update_interval(self, pca_test_data, update_interval):
        """Test that update_interval triggers partial_fits correctly"""
        n_components = pca_test_data["n_components"]
        message = pca_test_data["message"]

        # Create a transformer with update interval
        settings = IncrementalDecompSettings(
            axis="feature",
            n_components=n_components,
            method="pca",
            update_interval=update_interval,  # Set update interval
        )
        transformer = IncrementalDecompTransformer(settings=settings)

        # Create a spy on the partial_fit method to track calls
        original_partial_fit = transformer._procs["decomp"].partial_fit
        call_count = [0]

        def spy_partial_fit(msg):
            call_count[0] += 1
            return original_partial_fit(msg)

        transformer._procs["decomp"].partial_fit = spy_partial_fit

        # Process the message
        _ = transformer(message)

        # Check that partial_fit was called an appropriate number of times.
        n_samples = message.data.shape[message.get_axis_idx("time")]
        n_per_partial_fit = int(update_interval / message.axes["time"].gain)
        assert call_count[0] == int(n_samples / n_per_partial_fit)

    def test_different_axis(self, pca_test_data):
        """Test with different axis configurations"""
        message = pca_test_data["message"]
        n_components = pca_test_data["n_components"]

        # Test with !time axis
        settings = IncrementalDecompSettings(
            axis="!time",  # Decompose across all axes except time
            n_components=n_components,
            method="pca",
            update_interval=0.0,
        )
        transformer = IncrementalDecompTransformer(settings=settings)

        # First do partial_fit manually to ensure the model is trained
        transformer._procs["decomp"].partial_fit(message)

        # Then process
        result = transformer(message)

        # Check that the output has the expected dimensions
        assert isinstance(result, AxisArray)
        assert "time" in result.dims
        assert "components" in result.dims
        assert result.data.shape == (message.data.shape[0], n_components)

    def test_pca_stateful_op(self, pca_test_data):
        message = pca_test_data["message"]
        n_components = pca_test_data["n_components"]
        settings = IncrementalDecompSettings(
            axis="!time",  # Decompose across all axes except time
            n_components=n_components,
            method="pca",
            update_interval=0.5,
        )
        transformer = IncrementalDecompTransformer(settings=settings)

        state1, res1 = transformer.stateful_op(None, message)
        assert "decomp" in state1
        estim_state = state1["decomp"][0].estimator
        assert (
            hasattr(estim_state, "components_") and estim_state.components_ is not None
        )
