from dataclasses import field

import ezmsg.core as ez
import numpy as np
from ezmsg.baseproc import (
    BaseAdaptiveTransformer,
    BaseAdaptiveTransformerUnit,
    processor_state,
)
from ezmsg.sigproc.util.segment import ConfigChain
from ezmsg.util.messages.axisarray import AxisArray, compute_config_id, replace

from .._optional import missing_extra
from ..util import RegressorType, StaticLinearRegressor, get_regressor

try:
    from sklearn.linear_model._base import LinearModel
except ImportError as exc:
    raise missing_extra("sklearn", __name__) from exc


class LinearRegressorSettings(ez.Settings):
    model_type: StaticLinearRegressor = StaticLinearRegressor.LINEAR
    settings_path: str | None = None
    model_kwargs: dict = field(default_factory=dict)


@processor_state
class LinearRegressorState:
    template: AxisArray | None = None
    model: LinearModel | None = None
    configs: ConfigChain | None = None
    """Derives the output's config_id from the input's and ``template``."""


class LinearRegressorTransformer(
    BaseAdaptiveTransformer[LinearRegressorSettings, AxisArray, AxisArray, LinearRegressorState]
):
    """
    Linear regressor.

    Note: `partial_fit` is not 'partial'. It fully resets the model using the entirety of the
    SampleTriggerMessage in the input message's "training" attrs.
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
            regressor_klass = get_regressor(RegressorType.STATIC, self.settings.model_type)
            self.state.model = regressor_klass(**self.settings.model_kwargs)

    def _reset_state(self, message: AxisArray) -> None:
        # So far, there is nothing to reset.
        #  .model and .template are initialized in __init__
        pass

    def partial_fit(self, message: AxisArray) -> None:
        if np.any(np.isnan(message.data)):
            return

        X = message.data
        y = message.attrs["trigger"].value.data
        # TODO: Resample should provide identical durations.
        self.state.model = self.state.model.fit(X[: y.shape[0]], y[: X.shape[0]])
        self.state.template = replace(
            message.attrs["trigger"].value,
            data=np.array([[]]),
            key=message.attrs["trigger"].value.key + "_pred",
        )
        self.state.configs = ConfigChain(compute_config_id(self.state.template))

    def _process(self, message: AxisArray) -> AxisArray:
        if self.state.template is None:
            return AxisArray(np.array([[]]), dims=["time", "ch"])
        if message.data.shape[0]:
            preds = self.state.model.predict(message.data)
        else:
            # sklearn will not predict for no samples (e.g. an end-of-stream marker).
            preds = np.zeros((0, *np.shape(self.state.model.coef_)[:-1]))
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
            stream_dim="time",
            segment=message.segment,
            config_id=self.state.configs.update(message.config_id),
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
