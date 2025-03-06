from dataclasses import field

import numpy as np
import pandas as pd
import river.optim
import river.linear_model
import sklearn.base
from ezmsg.sigproc.sampler import SampleMessage
from ezmsg.sigproc.base import (
    processor_settings,
    processor_state,
    BaseAdaptiveTransformer,
    BaseAdaptiveTransformerUnit,
)
from ezmsg.util.messages.axisarray import AxisArray, replace

from ..util import REGRESSORS, AdaptiveLinearRegressor


@processor_settings
class AdaptiveLinearRegressorSettings:
    model_type: AdaptiveLinearRegressor = AdaptiveLinearRegressor.LINEAR
    settings_path: str | None = None
    model_kwargs: dict = field(default_factory=dict)


@processor_state
class AdaptiveLinearRegressorState:
    template: AxisArray | None = None
    model: river.linear_model.base.GLM | sklearn.base.RegressorMixin | None = None


class AdaptiveLinearRegressorTransformer(
    BaseAdaptiveTransformer[
        AdaptiveLinearRegressorSettings, AxisArray, AxisArray, AdaptiveLinearRegressorState
    ]
):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        b_river = self.settings.model_type in [
            AdaptiveLinearRegressor.LINEAR,
            AdaptiveLinearRegressor.LOGISTIC,
        ]
        if b_river:
            self.settings.model_kwargs["l2"] = self.settings.model_kwargs.get("l2", 0.0)
            if "learn_rate" in self.settings.model_kwargs:
                self.settings.model_kwargs["optimizer"] = river.optim.SGD(
                    self.settings.model_kwargs.pop("learn_rate")
                )

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
            self.state.model = REGRESSORS[self.settings.model_type](
                **self.settings.model_kwargs
            )

    def _hash_message(self, message: AxisArray) -> int:
        # So far, nothing to reset so hash can be constant.
        return -1

    def _reset_state(self, message: AxisArray) -> None:
        # So far, there is nothing to reset.
        #  .model is initialized in __init__
        #  .template is updated in partial_fit
        pass

    def partial_fit(self, message: SampleMessage) -> None:
        if np.any(np.isnan(message.sample.data)):
            return

        if self.settings.model_type in [
            AdaptiveLinearRegressor.LINEAR,
            AdaptiveLinearRegressor.LOGISTIC,
        ]:
            x = pd.DataFrame.from_dict(
                {
                    k: v
                    for k, v in zip(
                        message.sample.axes["ch"].data, message.sample.data.T
                    )
                }
            )
            y = pd.Series(
                data=message.trigger.value.data[:, 0],
                name=message.trigger.value.axes["ch"].data[0],
            )
            self.state.model.learn_many(x, y)
        else:
            print("TODO: Do sklearn partial_fit")

        self.state.template = replace(
            message.trigger.value,
            data=np.array([]),
            key=message.trigger.value.key + "_pred",
        )

    def _process(self, message: AxisArray) -> AxisArray | None:
        if self.state.template is None:
            return AxisArray(np.array([]), dims=[""])

        if not np.any(np.isnan(message.data)):
            if self.settings.model_type in [
                AdaptiveLinearRegressor.LINEAR,
                AdaptiveLinearRegressor.LOGISTIC,
            ]:
                # TODO: covnert msg_in.data to something appropriate for river
                preds = self.state.model.predict_many(message.data)
                # TODO: Convert preds to a numpy array
            else:
                preds = self.state.model.predict(message.data)
            return replace(
                self.state.template,
                data=preds,
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
        AxisArray, AxisArray,
        AdaptiveLinearRegressorTransformer,
    ]
):
    SETTINGS = AdaptiveLinearRegressorSettings
