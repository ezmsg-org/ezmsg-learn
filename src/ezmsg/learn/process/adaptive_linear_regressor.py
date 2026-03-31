"""Adaptive linear regressor processor.

.. note::
    This module supports the Array API standard via
    ``array_api_compat.get_namespace()``.  NaN checks and axis permutations
    use Array API operations; a NumPy boundary is applied before sklearn
    ``partial_fit``/``predict`` and before river ``learn_many``/``predict_many``.
"""

import copy
import typing
from dataclasses import field

import ezmsg.core as ez
import numpy as np
import pandas as pd
import river.linear_model
import river.optim
import sklearn.base
from array_api_compat import get_namespace, is_numpy_array
from ezmsg.baseproc import (
    BaseAdaptiveTransformer,
    BaseAdaptiveTransformerUnit,
    processor_state,
)
from ezmsg.util.messages.axisarray import AxisArray, replace

from ..util import AdaptiveLinearRegressor, RegressorType, get_regressor


class AdaptiveLinearRegressorSettings(ez.Settings):
    model_type: AdaptiveLinearRegressor = AdaptiveLinearRegressor.LINEAR
    settings_path: str | None = None
    model_kwargs: dict = field(default_factory=dict)


@processor_state
class AdaptiveLinearRegressorState:
    template: AxisArray | None = None
    model: (
        river.linear_model.base.GLM
        | dict[typing.Hashable, river.linear_model.base.GLM]
        | sklearn.base.RegressorMixin
        | None
    ) = None


def _normalize_axis_label(label):
    dtype_names = getattr(getattr(label, "dtype", None), "names", None)
    if dtype_names is not None:
        if "label" in dtype_names:
            return str(label["label"])
        return tuple((name, _normalize_axis_label(label[name])) for name in dtype_names)

    if isinstance(label, np.generic):
        return label.item()

    try:
        hash(label)
        return label
    except TypeError:
        return str(label)


def _axis_labels(axis_data) -> list:
    return [_normalize_axis_label(label) for label in axis_data]


def _prediction_template(message: AxisArray) -> AxisArray:
    return replace(
        message,
        data=np.empty_like(message.data),
        key=message.key + "_pred",
    )


def _prediction_template_from_signal(message: AxisArray, output_labels: list[typing.Hashable]) -> AxisArray:
    n_time = message.data.shape[message.get_axis_idx("time")]
    return AxisArray(
        data=np.empty((n_time, len(output_labels))),
        dims=["time", "ch"],
        axes={
            "time": replace(message.axes["time"], offset=message.axes["time"].offset),
            "ch": AxisArray.CoordinateAxis(data=np.asarray(output_labels), dims=["ch"]),
        },
        key=message.key + "_pred",
    )


def _output_labels(message: AxisArray) -> list[typing.Hashable]:
    if "ch" not in message.axes:
        data = np.asarray(message.data)
        width = data.shape[-1] if data.ndim > 1 else 1
        return [f"ch{idx}" for idx in range(width)]
    return _axis_labels(message.axes["ch"].data)


class AdaptiveLinearRegressorTransformer(
    BaseAdaptiveTransformer[
        AdaptiveLinearRegressorSettings,
        AxisArray,
        AxisArray,
        AdaptiveLinearRegressorState,
    ]
):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        model_kwargs = dict(self.settings.model_kwargs)
        model_type = AdaptiveLinearRegressor(self.settings.model_type)
        b_river = self.settings.model_type in [
            AdaptiveLinearRegressor.LINEAR,
            AdaptiveLinearRegressor.LOGISTIC,
        ]
        if b_river:
            model_kwargs.setdefault("l2", 0.0)
            if "learn_rate" in model_kwargs:
                model_kwargs["optimizer"] = river.optim.SGD(model_kwargs.pop("learn_rate"))

        self.settings = replace(
            self.settings,
            model_type=model_type,
            model_kwargs=model_kwargs,
        )
        self._regressor_klass = get_regressor(RegressorType.ADAPTIVE, self.settings.model_type)
        if self.settings.settings_path is not None:
            # Load model from file
            import pickle

            with open(self.settings.settings_path, "rb") as f:
                model = pickle.load(f)

            if b_river:
                models = model.values() if isinstance(model, dict) else [model]
                for river_model in models:
                    river_model.l2 = self.settings.model_kwargs["l2"]
                    if "optimizer" in self.settings.model_kwargs:
                        river_model.optimizer = copy.deepcopy(self.settings.model_kwargs["optimizer"])
            else:
                print("TODO: Override sklearn model with kwargs")
            self.state.model = model
        elif not b_river:
            # Build model from scratch.
            self.state.model = self._regressor_klass(**self.settings.model_kwargs)

    def _hash_message(self, message: AxisArray) -> int:
        # So far, nothing to reset so hash can be constant.
        return -1

    def _reset_state(self, message: AxisArray) -> None:
        # So far, there is nothing to reset.
        #  .model is initialized in __init__
        #  .template is updated in partial_fit
        pass

    def _prediction_labels(self, n_outputs: int) -> list[typing.Hashable]:
        if self.state.template is not None:
            return _output_labels(self.state.template)
        if isinstance(self.state.model, dict):
            return list(self.state.model.keys())
        return [f"ch{idx}" for idx in range(n_outputs)]

    def partial_fit(self, message: AxisArray) -> None:
        xp = get_namespace(message.data)

        if xp.any(xp.isnan(message.data)):
            return

        if self.settings.model_type in [
            AdaptiveLinearRegressor.LINEAR,
            AdaptiveLinearRegressor.LOGISTIC,
        ]:
            # river path: needs numpy/pandas
            data_np = np.asarray(message.data) if not is_numpy_array(message.data) else message.data
            x = pd.DataFrame(data_np, columns=_axis_labels(message.axes["ch"].data))
            targets = message.attrs["trigger"].value
            target_np = np.asarray(targets.data)
            if target_np.ndim == 1:
                target_np = target_np[:, None]
            target_labels = _output_labels(targets)
            if self.state.model is None:
                if len(target_labels) == 1:
                    self.state.model = self._regressor_klass(**copy.deepcopy(self.settings.model_kwargs))
                else:
                    models = {}
                    for label in target_labels:
                        models[label] = self._regressor_klass(**copy.deepcopy(self.settings.model_kwargs))
                    self.state.model = {label: models[label] for label in target_labels}
            models = self.state.model
            if len(target_labels) == 1 and not isinstance(models, dict):
                models = {target_labels[0]: models}
            if set(target_labels) != set(models.keys()):
                ez.logger.error(f"Target labels ({target_labels}) does not match model labels ({list(models.keys())}).")
                raise ValueError("Target labels do not match model labels.")
            for idx, label in enumerate(target_labels):
                y = pd.Series(data=target_np[:, idx], name=label)
                models[label].learn_many(x, y)
        else:
            # sklearn path: permute then convert to numpy
            X = message.data
            ax_idx = message.get_axis_idx("time")
            if ax_idx != 0:
                perm = (ax_idx,) + tuple(i for i in range(X.ndim) if i != ax_idx)
                X = xp.permute_dims(X, perm)
            X_np = np.asarray(X) if not is_numpy_array(X) else X
            self.state.model.partial_fit(X_np, message.attrs["trigger"].value.data)

        self.state.template = _prediction_template(message.attrs["trigger"].value)

    def _process(self, message: AxisArray) -> AxisArray | None:
        if self.state.model is None:
            return AxisArray(np.array([]), dims=[""])

        xp = get_namespace(message.data)

        if not xp.any(xp.isnan(message.data)):
            if self.settings.model_type in [
                AdaptiveLinearRegressor.LINEAR,
                AdaptiveLinearRegressor.LOGISTIC,
            ]:
                # river path: needs numpy/pandas
                data_np = np.asarray(message.data) if not is_numpy_array(message.data) else message.data
                x = pd.DataFrame(data_np, columns=_axis_labels(message.axes["ch"].data))
                n_outputs = len(self.state.model) if isinstance(self.state.model, dict) else 1
                out_labels = self._prediction_labels(n_outputs)
                if isinstance(self.state.model, dict):
                    pred_cols = []
                    for label in out_labels:
                        model = self.state.model.get(label)
                        if model is None:
                            pred_cols.append(np.zeros(len(x), dtype=float))
                        else:
                            pred_cols.append(model.predict_many(x).to_numpy())
                    preds = np.column_stack(pred_cols)
                else:
                    first_col = self.state.model.predict_many(x).to_numpy()
                    if len(out_labels) == 1:
                        preds = first_col[:, None]
                    else:
                        zeros = np.zeros((len(x), len(out_labels) - 1), dtype=float)
                        preds = np.column_stack([first_col, zeros])
            else:
                # sklearn path: needs numpy
                data_np = np.asarray(message.data) if not is_numpy_array(message.data) else message.data
                preds = self.state.model.predict(data_np)
            preds = preds.reshape((len(preds), -1))
            template = self.state.template
            if template is None:
                template = _prediction_template_from_signal(message, self._prediction_labels(preds.shape[1]))
                self.state.template = template
            return replace(
                template,
                data=preds,
                axes={
                    **template.axes,
                    "time": message.axes["time"],
                },
            )


class AdaptiveLinearRegressorUnit(
    BaseAdaptiveTransformerUnit[
        AdaptiveLinearRegressorSettings,
        AxisArray,
        AxisArray,
        AdaptiveLinearRegressorTransformer,
    ]
):
    SETTINGS = AdaptiveLinearRegressorSettings
