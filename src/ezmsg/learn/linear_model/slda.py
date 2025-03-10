import typing

import ezmsg.core as ez
import numpy as np
import numpy.typing as npt
import scipy.io as sio
from ezmsg.util.generator import consumer
from ezmsg.util.messages.axisarray import AxisArray, slice_along_axis
from ezmsg.util.messages.util import replace
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

from ..util import ClassifierMessage


@consumer
def slda_decoder(
    settings_path: str, axis: str = "time"
) -> typing.Generator[ClassifierMessage, AxisArray, None]:
    msg_out = ClassifierMessage(data=np.array([]), dims=[""])  # Placeholder
    channels: typing.Union[slice, npt.NDArray] = np.s_[:]
    b_from_mat = settings_path[-4:] == ".mat"

    if b_from_mat:
        # Expects a very specific format from a specific project. Not for general use.
        matlab_sLDA = sio.loadmat(settings_path, squeeze_me=True)
        weights = matlab_sLDA["weights"][1, 1:]
        # mean = matlab_sLDA['mXtrain']
        # std = matlab_sLDA['sXtrain']
        # lags = matlab_sLDA['lags'] + 1
        channels = matlab_sLDA["channels"] - 4
        channels -= channels[0]  # Offsets are wrong somehow.
        lda = LDA(solver="lsqr", shrinkage="auto")
        lda.classes_ = np.asarray([0, 1])
        lda.coef_ = np.expand_dims(weights, axis=0)
        lda.intercept_ = matlab_sLDA["weights"][1, 0]
    else:
        import pickle

        with open(settings_path, "rb") as f:
            lda = pickle.load(f)

    out_template: typing.Optional[ClassifierMessage] = None
    samp_ax_idx: int = 0
    ch_ax_idx: int = -1

    while True:
        axis_arr_in = yield msg_out

        if out_template is None:
            # Create template ClassifierMessage using first msg and lda.classes_
            out_labels = lda.classes_.tolist()
            zero_shape = (0, len(out_labels))
            out_template = ClassifierMessage(
                data=np.zeros(zero_shape, dtype=axis_arr_in.data.dtype),
                dims=[axis, "classes"],
                axes={
                    axis: axis_arr_in.axes[axis],
                    "classes": AxisArray.CoordinateAxis(
                        data=out_labels, dims=["classes"]
                    ),
                },
                labels=out_labels,
                key=axis_arr_in.key,
            )
            ch_ax_idx = axis_arr_in.dims.index("ch")
            samp_ax_idx = axis_arr_in.dims.index(axis)

        X = slice_along_axis(axis_arr_in.data, channels, ch_ax_idx)
        X = np.moveaxis(X, samp_ax_idx, 0)
        if X.shape[0]:
            if b_from_mat:
                pred_probas = []
                for samp in X:
                    tmp = samp.flatten(order="F") * 1e-6
                    tmp = np.expand_dims(tmp, axis=0)
                    # tmp = (tmp - mean) / std
                    probas = lda.predict_proba(tmp)
                    pred_probas.append(probas)
                pred_probas = np.concatenate(pred_probas, axis=0)
            else:
                # This creates a copy.
                X = X.reshape(X.shape[0], -1)
                pred_probas = lda.predict_proba(X)

            update_ax = out_template.axes[axis]
            update_ax.offset = axis_arr_in.axes[axis].offset
            msg_out = replace(
                out_template,
                data=pred_probas,
                axes={
                    **out_template.axes,
                    # `replace` will copy the minimal set of fields
                    axis: replace(update_ax, offset=update_ax.offset),
                },
            )
        else:
            msg_out = out_template


class SLDASettings(ez.Settings):
    settings_path: str
    axis: str = "time"


class SLDAState(ez.State):
    gen: typing.Generator[ClassifierMessage, AxisArray, None]


class SLDA(ez.Unit):
    SETTINGS = SLDASettings
    STATE = SLDAState

    INPUT_SIGNAL = ez.InputStream(AxisArray)
    OUTPUT = ez.OutputStream(ClassifierMessage)

    def construct_generator(self):
        self.STATE.gen = slda_decoder(
            settings_path=self.SETTINGS.settings_path, axis=self.SETTINGS.axis
        )

    def initialize(self):
        self.construct_generator()

    @ez.subscriber(INPUT_SIGNAL)
    @ez.publisher(OUTPUT)
    async def on_message(self, message: AxisArray):
        out_msg = self.STATE.gen.send(message)
        if out_msg.data.size:
            yield self.OUTPUT, out_msg
