import importlib
import pickle
import typing

import ezmsg.core as ez
import numpy as np
import pandas as pd
from ezmsg.sigproc.base import (
    BaseAdaptiveTransformer,
    BaseAdaptiveTransformerUnit,
    processor_state,
)
from ezmsg.sigproc.sampler import SampleMessage
from ezmsg.util.messages.axisarray import AxisArray
from ezmsg.util.messages.util import replace


class SklearnModelSettings(ez.Settings):
    model_class: str
    """
    Full path to the sklearn model class
    Example: 'sklearn.linear_model.LinearRegression'
    """
    model_kwargs: dict[str, typing.Any] = None
    """
    Additional keyword arguments to pass to the model constructor.
    Example: {'fit_intercept': True, 'normalize': False}
    """
    checkpoint_path: str | None = None
    """
    Path to a checkpoint file to load the model from.
    If provided, the model will be initialized from this checkpoint.
    Example: 'path/to/checkpoint.pkl'
    """
    partial_fit_classes: np.ndarray | None = None
    """
    The full list of classes to use for partial_fit calls.
    This must be provided to use `partial_fit` with classifiers.
    """


@processor_state
class SklearnModelState:
    model: typing.Any = None
    chan_ax: AxisArray.CoordinateAxis | None = None


class SklearnModelProcessor(
    BaseAdaptiveTransformer[
        SklearnModelSettings, AxisArray, AxisArray, SklearnModelState
    ]
):
    """
    Processor that wraps a scikit-learn, River, or HMMLearn model for use in the ezmsg framework.

    This processor supports:
    - `fit`, `partial_fit`, or River's `learn_many`/`learn_one` for training.
    - `predict`, River's `predict_many`, or `predict_one` for inference.
    - Optional model checkpoint loading and saving.

    The processor expects and outputs `AxisArray` messages with a `"ch"` (channel) axis.

    Settings:
    ---------
    model_class : str
        Full path to the sklearn or River model class to use.
        Example: "sklearn.linear_model.SGDClassifier" or "river.linear_model.LogisticRegression"

    model_kwargs : dict[str, typing.Any], optional
        Additional keyword arguments passed to the model constructor.

    checkpoint_path : str, optional
        Path to a pickle file to load a previously saved model. If provided, the model will
        be restored from this path at startup.

    partial_fit_classes : np.ndarray, optional
        For classifiers that require all class labels to be specified during `partial_fit`.

    Example:
    -----------------------------
    ```python
    processor = SklearnModelProcessor(
        settings=SklearnModelSettings(
            model_class='sklearn.linear_model.SGDClassifier',
            model_kwargs={'loss': 'log_loss'},
            partial_fit_classes=np.array([0, 1]),
        )
    )
    ```
    """

    def _init_model(self) -> None:
        module_path, class_name = self.settings.model_class.rsplit(".", 1)
        model_cls = getattr(importlib.import_module(module_path), class_name)
        kwargs = self.settings.model_kwargs or {}
        self._state.model = model_cls(**kwargs)

    def save_checkpoint(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump(self._state.model, f)

    def load_checkpoint(self, path: str) -> None:
        try:
            with open(path, "rb") as f:
                self._state.model = pickle.load(f)
        except Exception as e:
            ez.logger.error(f"Failed to load model from {path}: {str(e)}")
            raise RuntimeError(f"Failed to load model from {path}: {str(e)}") from e

    def _reset_state(self, message: AxisArray) -> None:
        # Try loading from checkpoint first
        if self.settings.checkpoint_path:
            self.load_checkpoint(self.settings.checkpoint_path)
            n_input = message.data.shape[message.get_axis_idx("ch")]
            if hasattr(self._state.model, "n_features_in_"):
                expected = self._state.model.n_features_in_
                if expected != n_input:
                    raise ValueError(
                        f"Model expects {expected} features, but got {n_input}"
                    )
        else:
            # No checkpoint, initialize from scratch
            self._init_model()

    def partial_fit(self, message: SampleMessage) -> None:
        X = message.sample.data
        y = message.trigger.value
        if self._state.model is None:
            self._reset_state(message.sample)
        if hasattr(self._state.model, "partial_fit"):
            kwargs = {}
            if self.settings.partial_fit_classes is not None:
                kwargs["classes"] = self.settings.partial_fit_classes
            self._state.model.partial_fit(X, y, **kwargs)
        elif hasattr(self._state.model, "learn_many"):
            df_X = pd.DataFrame(
                {
                    k: v
                    for k, v in zip(
                        message.sample.axes["ch"].data, message.sample.data.T
                    )
                }
            )
            name = (
                message.trigger.value.axes["ch"].data[0]
                if hasattr(message.trigger.value, "axes")
                and "ch" in message.trigger.value.axes
                else "target"
            )
            ser_y = pd.Series(
                data=np.asarray(message.trigger.value.data).flatten(),
                name=name,
            )
            self._state.model.learn_many(df_X, ser_y)
        elif hasattr(self._state.model, "learn_one"):
            # river's random forest does not support learn_many
            for xi, yi in zip(X, y):
                features = {f"f{i}": xi[i] for i in range(len(xi))}
                self._state.model.learn_one(features, yi)
        else:
            raise NotImplementedError(
                "Model does not support partial_fit or learn_many"
            )

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        if self._state.model is None:
            dummy_msg = AxisArray(
                data=X,
                dims=["time", "ch"],
                axes={
                    "time": AxisArray.TimeAxis(fs=1.0),
                    "ch": AxisArray.CoordinateAxis(
                        data=np.array([f"ch_{i}" for i in range(X.shape[1])]),
                        dims=["ch"],
                    ),
                },
            )
            self._reset_state(dummy_msg)
        if hasattr(self._state.model, "fit"):
            self._state.model.fit(X, y)
        elif hasattr(self._state.model, "learn_many"):
            df_X = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
            ser_y = pd.Series(y.flatten(), name="target")
            self._state.model.learn_many(df_X, ser_y)
        elif hasattr(self._state.model, "learn_one"):
            # river's random forest does not support learn_many
            for xi, yi in zip(X, y):
                features = {f"f{i}": xi[i] for i in range(len(xi))}
                self._state.model.learn_one(features, yi)
        else:
            raise NotImplementedError("Model does not support fit or learn_many")

    def _process(self, message: AxisArray) -> AxisArray:
        if self._state.model is None:
            raise RuntimeError(
                "Model has not been fit yet. Call `fit()` or `partial_fit()` before processing."
            )
        X = message.data
        original_shape = X.shape
        n_input = X.shape[message.get_axis_idx("ch")]

        # Ensure X is 2D
        X = X.reshape(-1, n_input)
        if hasattr(self._state.model, "n_features_in_"):
            expected = self._state.model.n_features_in_
            if expected != n_input:
                raise ValueError(
                    f"Model expects {expected} features, but got {n_input}"
                )

        if hasattr(self._state.model, "predict"):
            y_pred = self._state.model.predict(X)
        elif hasattr(self._state.model, "predict_many"):
            df_X = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
            y_pred = self._state.model.predict_many(df_X)
            y_pred = np.array(list(y_pred))
        elif hasattr(self._state.model, "predict_one"):
            # river's random forest does not support predict_many
            y_pred = np.array(
                [
                    self._state.model.predict_one(
                        {f"f{i}": xi[i] for i in range(len(xi))}
                    )
                    for xi in X
                ]
            )
        else:
            raise NotImplementedError("Model does not support predict or predict_many")

        # For scalar outputs, ensure the output is 2D
        if y_pred.ndim == 1:
            y_pred = y_pred[:, np.newaxis]

        output_shape = original_shape[:-1] + (y_pred.shape[-1],)
        y_pred = y_pred.reshape(output_shape)

        if self._state.chan_ax is None:
            self._state.chan_ax = AxisArray.CoordinateAxis(
                data=np.arange(output_shape[1]), dims=["ch"]
            )

        return replace(
            message,
            data=y_pred,
            axes={**message.axes, "ch": self._state.chan_ax},
        )


class SklearnModelUnit(
    BaseAdaptiveTransformerUnit[
        SklearnModelSettings, AxisArray, AxisArray, SklearnModelProcessor
    ]
):
    """
    Unit wrapper for the `SklearnModelProcessor`.

    This unit provides a plug-and-play interface for using a scikit-learn or River model
    in an ezmsg graph-based system. It takes in `AxisArray` inputs and outputs predictions
    in the same format, optionally performing training via `partial_fit` or `fit`.

    Example:
    --------
    ```python
    unit = SklearnModelUnit(
        settings=SklearnModelSettings(
            model_class='sklearn.linear_model.SGDClassifier',
            model_kwargs={'loss': 'log_loss'},
            partial_fit_classes=np.array([0, 1]),
        )
    )
    ```
    """

    SETTINGS = SklearnModelSettings
