import numpy as np
import pytest
from ezmsg.baseproc import SampleTriggerMessage
from ezmsg.util.messages.axisarray import AxisArray
from ezmsg.util.messages.util import replace

from ezmsg.learn.process.sklearn import SklearnModelProcessor


@pytest.fixture
def input_axisarray():
    data = np.random.randn(10, 4)
    return AxisArray(
        data=data,
        dims=["time", "ch"],
        axes={
            "time": AxisArray.TimeAxis(fs=100.0),
            "ch": AxisArray.CoordinateAxis(data=np.arange(4), dims=["ch"]),
        },
    )


@pytest.fixture
def labels_classification():
    return np.random.randint(0, 2, size=(10,))


@pytest.fixture
def labels_regression():
    return np.random.randn(10)


@pytest.mark.parametrize(
    "model_class,is_classifier",
    [
        ("river.linear_model.LinearRegression", False),
        ("river.linear_model.LogisticRegression", True),
        ("sklearn.linear_model.Ridge", False),
        ("sklearn.discriminant_analysis.LinearDiscriminantAnalysis", True),
    ],
)
def test_output_shape_inference(
    model_class,
    is_classifier,
    input_axisarray,
    labels_classification,
    labels_regression,
):
    proc = SklearnModelProcessor(model_class=model_class)
    proc._reset_state(input_axisarray)

    # Fit the model before prediction
    labels = labels_classification if is_classifier else labels_regression
    proc.fit(input_axisarray.data, labels)

    output = proc._process(input_axisarray)
    assert output.data.shape[0] == input_axisarray.data.shape[0]
    assert output.data.ndim == 2


@pytest.mark.parametrize(
    "model_class,is_classifier",
    [
        ("river.linear_model.LinearRegression", False),
        ("river.linear_model.LogisticRegression", True),
        ("sklearn.linear_model.SGDClassifier", True),
        ("sklearn.linear_model.SGDRegressor", False),
    ],
)
def test_partial_fit_supported_models(
    model_class,
    is_classifier,
    input_axisarray,
    labels_classification,
    labels_regression,
):
    labels = labels_classification if is_classifier else labels_regression

    settings_kwargs = {"model_class": model_class}

    if is_classifier:
        settings_kwargs["partial_fit_classes"] = np.array([0, 1])

    proc = SklearnModelProcessor(**settings_kwargs)
    proc._reset_state(input_axisarray)

    sample_msg = replace(
        input_axisarray,
        attrs={**input_axisarray.attrs, "trigger": SampleTriggerMessage(timestamp=0.0, value=labels)},
    )

    proc.partial_fit(sample_msg)
    output = proc._process(input_axisarray)
    assert output.data.shape[0] == input_axisarray.data.shape[0]


def test_partial_fit_unsupported_model(input_axisarray, labels_regression):
    proc = SklearnModelProcessor(model_class="sklearn.linear_model.Ridge")
    proc._reset_state(input_axisarray)
    sample_msg = replace(
        input_axisarray,
        attrs={**input_axisarray.attrs, "trigger": SampleTriggerMessage(timestamp=0.0, value=labels_regression)},
    )
    with pytest.raises(NotImplementedError, match="partial_fit"):
        proc.partial_fit(sample_msg)


def test_partial_fit_changes_model(input_axisarray, labels_regression):
    proc = SklearnModelProcessor(model_class="sklearn.linear_model.SGDRegressor")
    proc._reset_state(input_axisarray)

    sample_msg = replace(
        input_axisarray,
        attrs={**input_axisarray.attrs, "trigger": SampleTriggerMessage(timestamp=0.0, value=labels_regression)},
    )

    proc.partial_fit(sample_msg)
    output_before = proc._process(input_axisarray).data.copy()
    proc.partial_fit(sample_msg)
    output_after = proc._process(input_axisarray).data
    assert not np.allclose(output_before, output_after)


def test_model_save_and_load(tmp_path, input_axisarray):
    proc = SklearnModelProcessor(model_class="sklearn.linear_model.Ridge")
    proc._reset_state(input_axisarray)

    checkpoint_path = tmp_path / "model_checkpoint.pkl"
    proc.save_checkpoint(str(checkpoint_path))

    new_proc = SklearnModelProcessor(model_class="sklearn.linear_model.Ridge", checkpoint_path=str(checkpoint_path))
    new_proc._reset_state(input_axisarray)
    assert new_proc._state.model is not None


def test_input_shape_mismatch_raises(input_axisarray, labels_regression):
    proc = SklearnModelProcessor(model_class="sklearn.linear_model.Ridge")
    proc._reset_state(input_axisarray)

    # Fit the model first so `n_features_in_` is set
    proc.fit(input_axisarray.data, labels_regression)

    # Create mismatched message with 3 input channels instead of 4
    bad_data = np.random.randn(10, 3)
    bad_msg = AxisArray(
        data=bad_data,
        dims=["time", "ch"],
        axes={
            "time": AxisArray.TimeAxis(fs=100.0),
            "ch": AxisArray.CoordinateAxis(data=np.arange(3), dims=["ch"]),
        },
    )

    with pytest.raises(ValueError, match="Model expects .* features, but got .*"):
        proc._process(bad_msg)


@pytest.mark.parametrize(
    "model_class,is_classifier",
    [
        ("river.forest.ARFClassifier", True),
        ("river.forest.ARFRegressor", False),
    ],
)
def test_random_forest_inference_shape(
    model_class,
    is_classifier,
    input_axisarray,
    labels_classification,
    labels_regression,
):
    labels = labels_classification if is_classifier else labels_regression

    proc = SklearnModelProcessor(model_class=model_class)
    proc._reset_state(input_axisarray)

    # Simulate one-time fitting (not partial)
    proc.fit(input_axisarray.data, labels)

    output = proc._process(input_axisarray)
    expected_output_dim = 1

    assert output.data.shape == (input_axisarray.data.shape[0], expected_output_dim)


def test_random_forest_save_load(tmp_path, input_axisarray, labels_classification):
    proc = SklearnModelProcessor(
        model_class="river.forest.ARFClassifier",
    )
    proc._reset_state(input_axisarray)
    proc.fit(input_axisarray.data, labels_classification)

    ckpt_path = tmp_path / "rf_ckpt.pkl"
    proc.save_checkpoint(str(ckpt_path))

    # Load new processor
    new_proc = SklearnModelProcessor(
        model_class="river.forest.ARFClassifier",
        checkpoint_path=str(ckpt_path),
    )
    new_proc._reset_state(input_axisarray)
    assert new_proc._state.model is not None

    # Check outputs still work
    output = new_proc._process(input_axisarray)
    assert output.data.shape[0] == input_axisarray.data.shape[0]


def test_hmmlearn_gaussianhmm_predict(input_axisarray):
    # Ensure data has no NaNs and is usable for fitting
    X = input_axisarray.data

    proc = SklearnModelProcessor(
        model_class="hmmlearn.hmm.GaussianHMM",
        model_kwargs={"n_components": 2, "n_iter": 10},
    )

    # hmmlearn expects (n_samples, n_features), so we combine time axis
    proc._reset_state(input_axisarray)
    proc.fit(X, None)  # HMM doesn't use labels

    output = proc._process(input_axisarray)

    # Output should be a sequence of state predictions, shape (timesteps, 1)
    assert output.data.shape[0] == input_axisarray.data.shape[0]
    assert output.data.ndim == 2
    assert output.data.shape[1] == 1


@pytest.fixture
def labels_two_class():
    return np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])


@pytest.mark.parametrize(
    "model_class",
    [
        "sklearn.discriminant_analysis.LinearDiscriminantAnalysis",
        "sklearn.linear_model.LogisticRegression",
    ],
)
def test_predict_proba_output(model_class, input_axisarray, labels_two_class):
    proc = SklearnModelProcessor(model_class=model_class, predict_method="predict_proba")
    proc._reset_state(input_axisarray)
    proc.fit(input_axisarray.data, labels_two_class)

    output = proc._process(input_axisarray)

    n_classes = len(np.unique(labels_two_class))
    assert output.data.shape == (input_axisarray.data.shape[0], n_classes)
    assert np.all(output.data >= 0.0)
    np.testing.assert_allclose(output.data.sum(axis=1), 1.0, rtol=1e-6, atol=1e-6)
    # Output ch axis is relabeled to the model's classes_.
    np.testing.assert_array_equal(output.axes["ch"].data, proc._state.model.classes_)


def test_predict_proba_partial_fit_classifier(input_axisarray, labels_two_class):
    proc = SklearnModelProcessor(
        model_class="sklearn.linear_model.SGDClassifier",
        model_kwargs={"loss": "log_loss"},
        partial_fit_classes=np.array([0, 1]),
        predict_method="predict_proba",
    )
    proc._reset_state(input_axisarray)
    sample_msg = replace(
        input_axisarray,
        attrs={**input_axisarray.attrs, "trigger": SampleTriggerMessage(timestamp=0.0, value=labels_two_class)},
    )
    proc.partial_fit(sample_msg)

    output = proc._process(input_axisarray)
    assert output.data.shape == (input_axisarray.data.shape[0], 2)
    np.testing.assert_allclose(output.data.sum(axis=1), 1.0, rtol=1e-6, atol=1e-6)


def test_predict_method_default_unchanged(input_axisarray, labels_two_class):
    # Default predict_method must match an explicit predict_method="predict".
    proc_default = SklearnModelProcessor(model_class="sklearn.linear_model.LogisticRegression")
    proc_default._reset_state(input_axisarray)
    proc_default.fit(input_axisarray.data, labels_two_class)

    proc_explicit = SklearnModelProcessor(
        model_class="sklearn.linear_model.LogisticRegression", predict_method="predict"
    )
    proc_explicit._reset_state(input_axisarray)
    proc_explicit.fit(input_axisarray.data, labels_two_class)

    out_default = proc_default._process(input_axisarray)
    out_explicit = proc_explicit._process(input_axisarray)
    np.testing.assert_array_equal(out_default.data, out_explicit.data)
    assert out_default.data.shape == (input_axisarray.data.shape[0], 1)


def test_predict_proba_unsupported_model_raises(input_axisarray, labels_regression):
    proc = SklearnModelProcessor(model_class="sklearn.linear_model.Ridge", predict_method="predict_proba")
    proc._reset_state(input_axisarray)
    proc.fit(input_axisarray.data, labels_regression)
    with pytest.raises(NotImplementedError, match="predict_proba"):
        proc._process(input_axisarray)
