import pytest
import numpy as np

from ezmsg.util.messages.axisarray import AxisArray, slice_along_axis, replace
from ezmsg.learn.dim_reduce.incremental_decomp import (
    IncrementalDecompSettings,
    IncrementalDecompTransformer,
)
from ezmsg.learn.dim_reduce.adaptive_decomp import (
    IncrementalPCATransformer,
    MiniBatchNMFTransformer,
)


def gen_test_data(nmf: bool = False):
    # Create random data with predictable components
    fs = 50.0
    test_dur = 10.0
    chunk_dur = 0.2
    n_features = 10
    n_components = 3

    # Create data with clear components
    np.random.seed(42)
    n_samples = int(fs * test_dur)

    if nmf:
        # Non-negative data
        components = np.abs(np.random.randn(n_components, n_features))
        coefficients = np.abs(np.random.randn(n_samples, n_components))
        data = np.dot(coefficients, components)
    else:
        # Standard PCA data
        components = np.random.randn(n_components, n_features)
        coefficients = np.random.randn(n_samples, n_components)
        data = np.dot(coefficients, components)
        # Add noise
        data += 0.1 * np.random.randn(n_samples, n_features)

    # Reshape with empty channels dimension
    data = data.reshape(n_samples, 1, n_features)

    # Prepare chunking
    t0 = 0.0
    chunk0_dur = 2.0
    chunk0_len = int(chunk0_dur * fs)
    chunk_len = int(chunk_dur * fs)
    n_chunks = int((n_samples - chunk0_len) // chunk_len) + 1
    tvec = np.hstack(([0], np.arange(chunk0_len, n_samples, chunk_len))) / fs + t0

    template = AxisArray(
        slice_along_axis(data, slice(None, chunk0_len), 0),
        dims=["time", "ch", "feature"],
        axes={
            "time": AxisArray.TimeAxis(fs=fs, offset=t0),
            "ch": AxisArray.CoordinateAxis(
                data=np.array(["Ch0"]), unit="label", dims=["ch"]
            ),
        },
    )

    def axarr_generator():
        for chunk_ix in range(n_chunks):
            if chunk_ix == 0:
                # The first chunk is larger than the rest and is already in the template.
                axis_arr_out = template
            else:
                view = slice_along_axis(
                    data,
                    slice(
                        chunk0_len + (chunk_ix - 1) * chunk_len,
                        chunk0_len + chunk_ix * chunk_len,
                    ),
                    0,
                )
                axis_arr_out = replace(
                    template,
                    data=view,
                    axes={
                        **template.axes,
                        "time": replace(template.axes["time"], offset=tvec[chunk_ix]),
                    },
                )
            yield axis_arr_out

    return {
        "messages": axarr_generator(),
        "data": data,
        "n_components": n_components,
    }


@pytest.fixture
def pca_test_data():
    return gen_test_data(nmf=False)


@pytest.fixture
def nmf_test_data():
    return gen_test_data(nmf=True)


def _spy_partial_fit(transformer: IncrementalDecompTransformer, call_count: list[int]):
    """Spy on the partial_fit method to track calls"""
    original_partial_fit = transformer._procs["decomp"].partial_fit

    def spy_partial_fit(msg):
        call_count[0] += 1
        return original_partial_fit(msg)

    transformer._procs["decomp"].partial_fit = spy_partial_fit


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
        message_gen = test_data["messages"]

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

        # Spy on partial fit so we can check it was called exactly once.
        call_count = [0]
        _spy_partial_fit(transformer, call_count)

        result = [transformer(message) for message in message_gen]

        # Check output
        assert call_count[0] == 1, "Only first message should call partial_fit"
        assert isinstance(result[0], AxisArray)
        assert result[0].data.shape[1:] == (1, n_components)
        assert result[0].dims == ["time", "ch", "feature"]
        assert np.all(np.diff([msg.axes["time"].offset for msg in result]) > 0), (
            "Time axis offsets should be increasing"
        )
        if method == "nmf":
            assert np.all(result[0].data >= 0), "NMF output should be non-negative"

    @pytest.mark.parametrize("update_interval", [0.24, 0.5, 0.9, 1.0])
    def test_update_interval(self, pca_test_data, update_interval):
        """Test that update_interval triggers partial_fits correctly"""
        n_components = pca_test_data["n_components"]
        message_gen = pca_test_data["messages"]

        # Create a transformer with update interval
        settings = IncrementalDecompSettings(
            axis="feature",
            n_components=n_components,
            method="pca",
            update_interval=update_interval,  # Set update interval
        )
        transformer = IncrementalDecompTransformer(settings=settings)

        # Create a spy on the partial_fit method to track calls
        call_count = [0]
        _spy_partial_fit(transformer, call_count)

        # Process the messages
        result = [transformer(_) for _ in message_gen]

        # Check that partial_fit was called an appropriate number of times.
        # We know the input data is 10 seconds of 50 Hz data.
        #  We can use `update_interval` to calculate the expected number of partial_fit calls.
        n_calls_expected = 1 + int((10.0 - 2.0) / update_interval)
        assert call_count[0] == n_calls_expected
        assert isinstance(result[0], AxisArray)
        assert result[0].data.shape[1:] == (1, n_components)
        assert result[0].dims == ["time", "ch", "feature"]
        assert np.all(np.diff([msg.axes["time"].offset for msg in result]) > 0), (
            "Time axis offsets should be increasing"
        )

    def test_different_axis(self, pca_test_data):
        """Test with different axis configurations"""
        message_gen = pca_test_data["messages"]
        n_components = pca_test_data["n_components"]

        # Test with !time axis
        settings = IncrementalDecompSettings(
            axis="!time",  # Decompose across all axes except time
            n_components=n_components,
            method="pca",
            update_interval=0.0,
        )
        transformer = IncrementalDecompTransformer(settings=settings)
        result = [transformer(_) for _ in message_gen]

        # Check that the output has the expected dimensions
        assert isinstance(result[0], AxisArray)
        assert result[0].dims == ["time", "components"]
        assert result[0].data.shape[1:] == (n_components,)

    def test_pca_stateful_op(self, pca_test_data):
        message_gen = pca_test_data["messages"]
        n_components = pca_test_data["n_components"]
        settings = IncrementalDecompSettings(
            axis="!time",  # Decompose across all axes except time
            n_components=n_components,
            method="pca",
            update_interval=0.5,
        )
        transformer = IncrementalDecompTransformer(settings=settings)

        state = None
        result = []
        for msg in message_gen:
            state, _res = transformer.stateful_op(state, msg)
            result.append(_res)
            # Check the result
            assert isinstance(_res, AxisArray)
            assert _res.dims == ["time", "components"]
            assert _res.data.shape[1:] == (n_components,)
            # Check the state
            assert "decomp" in state
            estim_state = state["decomp"][0].estimator
            assert (
                hasattr(estim_state, "components_")
                and estim_state.components_ is not None
            )

        assert np.all(np.diff([msg.axes["time"].offset for msg in result]) > 0), (
            "Time axis offsets should be increasing"
        )
