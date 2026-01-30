import typing

import ezmsg.core as ez
import numpy as np
from ezmsg.baseproc import (
    BaseAdaptiveTransformer,
    BaseAdaptiveTransformerUnit,
    SampleMessage,
    processor_state,
)
from ezmsg.util.messages.axisarray import AxisArray
from ezmsg.util.messages.util import replace
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import SGDClassifier

from ..util import ClassifierMessage


class SGDDecoderSettings(ez.Settings):
    alpha: float = 1e-5
    eta0: float = 3e-4
    loss: str = "hinge"
    label_weights: dict[str, float] | None = None
    settings_path: str | None = None


@processor_state
class SGDDecoderState:
    model: typing.Any = None
    b_first_train: bool = True


class SGDDecoderTransformer(BaseAdaptiveTransformer[SGDDecoderSettings, AxisArray, ClassifierMessage, SGDDecoderState]):
    """
    SGD-based online classifier.

    Online Passive-Aggressive Algorithms
    <http://jmlr.csail.mit.edu/papers/volume7/crammer06a/crammer06a.pdf>
    K. Crammer, O. Dekel, J. Keshat, S. Shalev-Shwartz, Y. Singer - JMLR (2006)
    """

    def _refreshed_model(self):
        if self.settings.settings_path is not None:
            import pickle

            with open(self.settings.settings_path, "rb") as f:
                model = pickle.load(f)
                if self.settings.label_weights is not None:
                    model.class_weight = self.settings.label_weights
                model.eta0 = self.settings.eta0
        else:
            model = SGDClassifier(
                loss=self.settings.loss,
                alpha=self.settings.alpha,
                penalty="elasticnet",
                learning_rate="adaptive",
                eta0=self.settings.eta0,
                early_stopping=False,
                class_weight=self.settings.label_weights,
            )
        return model

    def _reset_state(self, message: AxisArray) -> None:
        self._state.model = self._refreshed_model()
        self._state.b_first_train = True

    def _process(self, message: AxisArray) -> ClassifierMessage | None:
        if self._state.model is None or not message.data.size:
            return None
        if np.any(np.isnan(message.data)):
            return None
        try:
            X = message.data.reshape((message.data.shape[0], -1))
            result = self._state.model._predict_proba_lr(X)
        except NotFittedError:
            return None
        out_axes = {}
        if message.dims[0] in message.axes:
            out_axes[message.dims[0]] = replace(
                message.axes[message.dims[0]],
                offset=message.axes[message.dims[0]].offset,
            )
        return ClassifierMessage(
            data=result,
            dims=message.dims[:1] + ["labels"],
            axes=out_axes,
            labels=list(self._state.model.class_weight.keys()),
            key=message.key,
        )

    def partial_fit(self, message: SampleMessage) -> None:
        if self._state.model is None:
            # Initialize model on first training sample
            self._state.model = self._refreshed_model()
            self._state.b_first_train = True

        if np.any(np.isnan(message.sample.data)):
            return
        train_sample = message.sample.data.reshape(1, -1)
        if self._state.b_first_train:
            self._state.model.partial_fit(
                train_sample,
                [message.trigger.value],
                classes=list(self.settings.label_weights.keys()),
            )
            self._state.b_first_train = False
        else:
            self._state.model.partial_fit(train_sample, [message.trigger.value])


class SGDDecoder(
    BaseAdaptiveTransformerUnit[
        SGDDecoderSettings,
        AxisArray,
        ClassifierMessage,
        SGDDecoderTransformer,
    ]
):
    SETTINGS = SGDDecoderSettings
