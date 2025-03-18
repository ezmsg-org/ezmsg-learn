import typing

import numpy as np
import ezmsg.core as ez
from ezmsg.util.messages.axisarray import AxisArray
from ezmsg.sigproc.base import (
    CompositeProcessor,
    BaseStatefulProcessor,
    BaseTransformerUnit,
)
from ezmsg.sigproc.window import WindowTransformer

from .adaptive_decomp import (
    IncrementalPCASettings,
    IncrementalPCATransformer,
    MiniBatchNMFSettings,
    MiniBatchNMFTransformer,
)


class IncrementalDecompSettings(ez.Settings):
    axis: str = "!time"
    n_components: int = 2
    update_interval: float = 0.0
    method: str = "pca"
    batch_size: typing.Optional[int] = None
    # PCA specific settings
    whiten: bool = False
    # NMF specific settings
    init: str = "random"
    beta_loss: str = "frobenius"
    tol: float = 1e-3
    alpha_W: float = 0.0
    alpha_H: typing.Union[float, str] = "same"
    l1_ratio: float = 0.0
    forget_factor: float = 0.7


class IncrementalDecompTransformer(
    CompositeProcessor[IncrementalDecompSettings, AxisArray, AxisArray]
):
    """
    Automates usage of IncrementalPCATransformer and MiniBatchNMFTransformer by using a WindowTransformer
    to extract training samples then calls partial_fit on the decomposition transformer.
    """

    @staticmethod
    def _initialize_processors(
        settings: IncrementalDecompSettings,
    ) -> dict[str, BaseStatefulProcessor]:
        # Create the appropriate decomposition transformer
        if settings.method == "pca":
            decomp_settings = IncrementalPCASettings(
                axis=settings.axis,
                n_components=settings.n_components,
                batch_size=settings.batch_size,
                whiten=settings.whiten,
            )
            decomp = IncrementalPCATransformer(settings=decomp_settings)
        else:  # nmf
            decomp_settings = MiniBatchNMFSettings(
                axis=settings.axis,
                n_components=settings.n_components,
                batch_size=settings.batch_size if settings.batch_size else 1024,
                init=settings.init,
                beta_loss=settings.beta_loss,
                tol=settings.tol,
                alpha_W=settings.alpha_W,
                alpha_H=settings.alpha_H,
                l1_ratio=settings.l1_ratio,
                forget_factor=settings.forget_factor,
            )
            decomp = MiniBatchNMFTransformer(settings=decomp_settings)

        # Create windowing processor if update_interval is specified
        if settings.update_interval > 0:
            # TODO: This `iter_axis` is likely incorrect.
            iter_axis = settings.axis[1:] if settings.axis.startswith("!") else "time"
            windowing = WindowTransformer(
                axis=iter_axis,
                window_dur=settings.update_interval,
                window_shift=settings.update_interval,
                zero_pad_until="none",
            )

            return {
                "decomp": decomp,
                "windowing": windowing,
            }

        return {"decomp": decomp}

    def stateful_op(
        self,
        state: dict[str, tuple[typing.Any, int]] | None,
        message: AxisArray,
    ) -> tuple[dict[str, tuple[typing.Any, int]], AxisArray]:
        state = state or {}

        estim = self._procs["decomp"]._state.estimator
        if not hasattr(estim, "components_") or estim.components_ is None:
            # If the estimator has not been trained once, train it with the first message
            self._procs["decomp"].partial_fit(message)
        elif "windowing" in self._procs:
            state["windowing"], train_msg = self._procs["windowing"].stateful_op(
                state.get("windowing", None), message
            )
            if np.prod(train_msg.data.shape) > 0:
                for _msg in train_msg.iter_over_axis("win"):
                    self._procs["decomp"].partial_fit(_msg)

        # Process the incoming message
        state["decomp"], result = self._procs["decomp"].stateful_op(
            state.get("decomp", None), message
        )

        return state, result

    def _process(self, message: AxisArray) -> AxisArray:
        estim = self._procs["decomp"]._state.estimator
        if not hasattr(estim, "components_") or estim.components_ is None:
            # If the estimator has not been trained once, train it with the first message
            self._procs["decomp"].partial_fit(message)
        elif "windowing" in self._procs:
            # If windowing is enabled, extract training samples and perform partial_fit
            train_msg = self._procs["windowing"](message)
            if np.prod(train_msg.data.shape) > 0:
                for _msg in train_msg.iter_over_axis("win"):
                    self._procs["decomp"].partial_fit(_msg)

        # Process the incoming message
        decomp_result = self._procs["decomp"](message)

        return decomp_result


class IncrementalDecompUnit(
    BaseTransformerUnit[
        IncrementalDecompSettings, AxisArray, AxisArray, IncrementalDecompTransformer
    ]
):
    SETTINGS = IncrementalDecompSettings
