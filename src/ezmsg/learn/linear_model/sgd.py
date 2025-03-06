import typing

import ezmsg.core as ez
import numpy as np
from ezmsg.sigproc.sampler import SampleMessage
from ezmsg.sigproc.base import GenAxisArray
from ezmsg.util.generator import consumer
from ezmsg.util.messages.axisarray import AxisArray
from ezmsg.util.messages.util import replace
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import SGDClassifier

from ..util import ClassifierMessage


@consumer
def sgd_decoder(
    alpha: float = 1.5e-5,
    eta0: float = 1e-7,  # Lower than what you'd use for offline training.
    loss: str = "squared_hinge",
    label_weights: dict[str, float] | None = None,
    settings_path: str | None = None,
) -> typing.Generator[AxisArray | SampleMessage, ClassifierMessage | None, None]:
    """
    Passive Aggressive Classifier
    Online Passive-Aggressive Algorithms <http://jmlr.csail.mit.edu/papers/volume7/crammer06a/crammer06a.pdf>
    K. Crammer, O. Dekel, J. Keshat, S. Shalev-Shwartz, Y. Singer - JMLR (2006)

    Args:
        alpha: Maximum step size (regularization)
        eta0: The initial learning rate for the 'adaptive’ schedules.
        loss: The loss function to be used:
            hinge: equivalent to PA-I in the reference paper.
            squared_hinge: equivalent to PA-II in the reference paper.
        label_weights: An optional dictionary of label names and their relative weight.
            e.g., {'Go': 31.0, 'Stop': 0.5}
            If this is None then settings_path must be provided and the pre-trained model
        settings_path: Path to the stored sklearn model pkl file.

    Returns:
        Generator that accepts `SampleMessage` for incremental training (`partial_fit`) and yields None,
        or `AxisArray` for inference (`predict`) and yields a `ClassifierMessage`.
    """
    # pre-init inputs and outputs
    msg_out = ClassifierMessage(data=np.array([]), dims=[""])

    # State variables:

    if settings_path is not None:
        import pickle

        with open(settings_path, "rb") as f:
            model = pickle.load(f)
            if label_weights is not None:
                model.class_weight = label_weights
            # Overwrite eta0, probably with a value lower than what was used online.
            model.eta0 = eta0
    else:
        model = SGDClassifier(
            loss=loss,
            alpha=alpha,
            penalty="elasticnet",
            learning_rate="adaptive",
            eta0=eta0,
            early_stopping=False,
            class_weight=label_weights,
        )

    b_first_train = True
    # TODO: template_out

    while True:
        msg_in: AxisArray | SampleMessage = yield msg_out

        msg_out = None
        if type(msg_in) is SampleMessage:
            # SampleMessage used for training.
            if not np.any(np.isnan(msg_in.sample.data)):
                train_sample = msg_in.sample.data.reshape(1, -1)
                if b_first_train:
                    model.partial_fit(
                        train_sample,
                        [msg_in.trigger.value],
                        classes=list(label_weights.keys()),
                    )
                    b_first_train = False
                else:
                    model.partial_fit(train_sample, [msg_in.trigger.value])
        elif msg_in.data.size:
            # AxisArray used for inference
            if not np.any(np.isnan(msg_in.data)):
                try:
                    X = msg_in.data.reshape((msg_in.data.shape[0], -1))
                    result = model._predict_proba_lr(X)
                except NotFittedError:
                    result = None
                if result is not None:
                    out_axes = {}
                    if msg_in.dims[0] in msg_in.axes:
                        out_axes[msg_in.dims[0]] = replace(
                            msg_in.axes[msg_in.dims[0]],
                            offset=msg_in.axes[msg_in.dims[0]].offset,
                        )
                    msg_out = ClassifierMessage(
                        data=result,
                        dims=msg_in.dims[:1] + ["labels"],
                        axes=out_axes,
                        labels=list(model.class_weight.keys()),
                        key=msg_in.key,
                    )


class SGDDecoderSettings(ez.Settings):
    alpha: float = 1e-5
    eta0: float = 3e-4
    loss: str = "hinge"
    label_weights: dict[str, float] | None = None
    settings_path: str | None = None


class SGDDecoder(GenAxisArray):
    SETTINGS = SGDDecoderSettings
    INPUT_SAMPLE = ez.InputStream(SampleMessage)

    # Method to be implemented by subclasses to construct the specific generator
    def construct_generator(self):
        self.STATE.gen = sgd_decoder(**self.SETTINGS.__dict__)

    @ez.subscriber(INPUT_SAMPLE)
    async def on_sample(self, msg: SampleMessage) -> None:
        _ = self.STATE.gen.send(msg)
