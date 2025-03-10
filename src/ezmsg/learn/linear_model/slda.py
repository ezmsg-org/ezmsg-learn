import typing

import ezmsg.core as ez
import numpy as np
from ezmsg.sigproc.base import (
    BaseStatefulTransformer,
    processor_state,
    BaseTransformerUnit,
)
from ezmsg.util.messages.axisarray import AxisArray
from ezmsg.util.messages.util import replace
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

from ..util import ClassifierMessage


class SLDASettings(ez.Settings):
    settings_path: str
    axis: str = "time"


@processor_state
class SLDAState:
    lda: LDA
    out_template: typing.Optional[ClassifierMessage] = None


class SLDATransformer(
    BaseStatefulTransformer[SLDASettings, AxisArray, ClassifierMessage, SLDAState]
):
    def _reset_state(self, message: AxisArray) -> None:
        if self.settings.settings_path[-4:] == ".mat":
            # Expects a very specific format from a specific project. Not for general use.
            import scipy.io as sio

            matlab_sLDA = sio.loadmat(self.settings.settings_path, squeeze_me=True)
            temp_weights = matlab_sLDA["weights"][1, 1:]
            temp_intercept = matlab_sLDA["weights"][1, 0]

            # Create weights and use zeros for channels we do not keep.
            channels = matlab_sLDA["channels"] - 4
            channels -= channels[0]  # Offsets are wrong somehow.
            n_channels = message.data.shape[message.dims.index("ch")]
            valid_indices = [ch for ch in channels if ch < n_channels]
            full_weights = np.zeros(n_channels)
            full_weights[valid_indices] = temp_weights[: len(valid_indices)]

            lda = LDA(solver="lsqr", shrinkage="auto")
            lda.classes_ = np.asarray([0, 1])
            lda.coef_ = np.expand_dims(full_weights, axis=0)
            lda.intercept_ = temp_intercept  # TODO: Is this supposed to be per-channel? Why the [1, 0]?
            self.state.lda = lda
            # mean = matlab_sLDA['mXtrain']
            # std = matlab_sLDA['sXtrain']
            # lags = matlab_sLDA['lags'] + 1
        else:
            import pickle

            with open(self.settings.settings_path, "rb") as f:
                self.state.lda = pickle.load(f)

        # Create template ClassifierMessage using lda.classes_
        out_labels = self.state.lda.classes_.tolist()
        zero_shape = (0, len(out_labels))
        self.state.out_template = ClassifierMessage(
            data=np.zeros(zero_shape, dtype=message.data.dtype),
            dims=[self.settings.axis, "classes"],
            axes={
                self.settings.axis: message.axes[self.settings.axis],
                "classes": AxisArray.CoordinateAxis(
                    data=np.array(out_labels), dims=["classes"]
                ),
            },
            labels=out_labels,
            key=message.key,
        )

    def _process(self, message: AxisArray) -> ClassifierMessage:
        samp_ax_idx = message.dims.index(self.settings.axis)
        X = np.moveaxis(message.data, samp_ax_idx, 0)

        if X.shape[0]:
            if (
                isinstance(self.settings.settings_path, str)
                and self.settings.settings_path[-4:] == ".mat"
            ):
                # Assumes F-contiguous weights
                pred_probas = []
                for samp in X:
                    tmp = samp.flatten(order="F") * 1e-6
                    tmp = np.expand_dims(tmp, axis=0)
                    probas = self.state.lda.predict_proba(tmp)
                    pred_probas.append(probas)
                pred_probas = np.concatenate(pred_probas, axis=0)
            else:
                # This creates a copy.
                X = X.reshape(X.shape[0], -1)
                pred_probas = self.state.lda.predict_proba(X)

            update_ax = self.state.out_template.axes[self.settings.axis]
            update_ax.offset = message.axes[self.settings.axis].offset

            return replace(
                self.state.out_template,
                data=pred_probas,
                axes={
                    **self.state.out_template.axes,
                    # `replace` will copy the minimal set of fields
                    self.settings.axis: replace(update_ax, offset=update_ax.offset),
                },
            )
        else:
            return self.state.out_template


class SLDA(
    BaseTransformerUnit[SLDASettings, AxisArray, ClassifierMessage, SLDATransformer]
):
    SETTINGS = SLDASettings
