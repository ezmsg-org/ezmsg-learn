from dataclasses import field

import numpy as np
from sklearn.linear_model._base import LinearModel
import ezmsg.core as ez
from ezmsg.sigproc.base import (
    processor_state,
    BaseAdaptiveTransformer,
    BaseAdaptiveTransformerUnit,
)
from ezmsg.util.messages.axisarray import AxisArray, replace
from ezmsg.sigproc.sampler import SampleMessage

from ..util import get_regressor, StaticLinearRegressor, RegressorType


class LinearRegressorSettings(ez.Settings):
    model_type: StaticLinearRegressor = StaticLinearRegressor.LINEAR
    settings_path: str | None = None
    model_kwargs: dict = field(default_factory=dict)


@processor_state
class LinearRegressorState:
    template: AxisArray | None = None
    model: LinearModel | None = None


class LinearRegressorTransformer(
    BaseAdaptiveTransformer[
        LinearRegressorSettings, AxisArray, AxisArray, LinearRegressorState
    ]
):
    """
    Linear regressor.

    Note: `partial_fit` is not 'partial'. It fully resets the model using the entirety of the SampleMessage provided.
    If you require adaptive fitting, try using the adaptive_linear_regressor module.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.settings.settings_path is not None:
            # Load model from file
            import pickle

            with open(self.settings.settings_path, "rb") as f:
                self.state.model = pickle.load(f)
        else:
            regressor_klass = get_regressor(
                RegressorType.STATIC, self.settings.model_type
            )
            self.state.model = regressor_klass(**self.settings.model_kwargs)

    def _reset_state(self, message: AxisArray) -> None:
        # So far, there is nothing to reset.
        #  .model and .template are initialized in __init__
        pass

    def partial_fit(self, message: SampleMessage) -> None:
        if np.any(np.isnan(message.sample.data)):
            return

        X = message.sample.data
        y = message.trigger.value.data
        # TODO: Resample should provide identical durations.
        self.state.model = self.state.model.fit(X[: y.shape[0]], y[: X.shape[0]])
        self.state.template = replace(
            message.trigger.value,
            data=np.array([[]]),
            key=message.trigger.value.key + "_pred",
        )

    def _process(self, message: AxisArray) -> AxisArray:
        if self.state.template is None:
            return AxisArray(np.array([[]]), dims=["time", "ch"])
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
        LinearRegressorSettings,
        AxisArray,
        AxisArray,
        LinearRegressorTransformer,
    ]
):
    SETTINGS = LinearRegressorSettings
