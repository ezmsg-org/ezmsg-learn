"""Adaptive linear regressor processor.

.. note::
    This module supports the Array API standard via
    ``array_api_compat.get_namespace()``.  NaN checks and axis permutations
    use Array API operations; a NumPy boundary is applied before sklearn
    ``partial_fit``/``predict`` and before river ``learn_many``/``predict_many``.
"""

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
    model: river.linear_model.base.GLM | sklearn.base.RegressorMixin | None = None


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
        self.settings = replace(self.settings, model_type=AdaptiveLinearRegressor(self.settings.model_type))
        b_river = self.settings.model_type in [
            AdaptiveLinearRegressor.LINEAR,
            AdaptiveLinearRegressor.LOGISTIC,
        ]
        if b_river:
            self.settings.model_kwargs["l2"] = self.settings.model_kwargs.get("l2", 0.0)
            if "learn_rate" in self.settings.model_kwargs:
                self.settings.model_kwargs["optimizer"] = river.optim.SGD(self.settings.model_kwargs.pop("learn_rate"))

        if self.settings.settings_path is not None:
            # Load model from file
            import pickle

            with open(self.settings.settings_path, "rb") as f:
                self.state.model = pickle.load(f)

            if b_river:
                # Override with kwargs?!
                self.state.model.l2 = self.settings.model_kwargs["l2"]
                if "optimizer" in self.settings.model_kwargs:
                    self.state.model.optimizer = self.settings.model_kwargs["optimizer"]
            else:
                print("TODO: Override sklearn model with kwargs")
        else:
            # Build model from scratch.
            regressor_klass = get_regressor(RegressorType.ADAPTIVE, self.settings.model_type)
            self.state.model = regressor_klass(**self.settings.model_kwargs)

    def _hash_message(self, message: AxisArray) -> int:
        # So far, nothing to reset so hash can be constant.
        return -1

    def _reset_state(self, message: AxisArray) -> None:
        # So far, there is nothing to reset.
        #  .model is initialized in __init__
        #  .template is updated in partial_fit
        pass

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
            x = pd.DataFrame.from_dict({k: v for k, v in zip(message.axes["ch"].data, data_np.T)})
            y = pd.Series(
                data=message.attrs["trigger"].value.data[:, 0],
                name=message.attrs["trigger"].value.axes["ch"].data[0],
            )
            self.state.model.learn_many(x, y)
        else:
            # sklearn path: permute then convert to numpy
            X = message.data
            ax_idx = message.get_axis_idx("time")
            if ax_idx != 0:
                perm = (ax_idx,) + tuple(i for i in range(X.ndim) if i != ax_idx)
                X = xp.permute_dims(X, perm)
            X_np = np.asarray(X) if not is_numpy_array(X) else X
            self.state.model.partial_fit(X_np, message.attrs["trigger"].value.data)

        self.state.template = replace(
            message.attrs["trigger"].value,
            data=np.empty_like(message.attrs["trigger"].value.data),
            key=message.attrs["trigger"].value.key + "_pred",
        )

    def _process(self, message: AxisArray) -> AxisArray | None:
        if self.state.template is None:
            return AxisArray(np.array([]), dims=[""])

        xp = get_namespace(message.data)

        if not xp.any(xp.isnan(message.data)):
            if self.settings.model_type in [
                AdaptiveLinearRegressor.LINEAR,
                AdaptiveLinearRegressor.LOGISTIC,
            ]:
                # river path: needs numpy/pandas
                data_np = np.asarray(message.data) if not is_numpy_array(message.data) else message.data
                x = pd.DataFrame.from_dict({k: v for k, v in zip(message.axes["ch"].data, data_np.T)})
                preds = self.state.model.predict_many(x).values
            else:
                # sklearn path: needs numpy
                data_np = np.asarray(message.data) if not is_numpy_array(message.data) else message.data
                preds = self.state.model.predict(data_np)
            return replace(
                self.state.template,
                data=preds.reshape((len(preds), -1)),
                axes={
                    **self.state.template.axes,
                    "time": replace(
                        message.axes["time"],
                        offset=message.axes["time"].offset,
                    ),
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
