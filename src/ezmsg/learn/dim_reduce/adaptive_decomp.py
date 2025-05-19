import typing

from sklearn.decomposition import IncrementalPCA, MiniBatchNMF
import numpy as np
import ezmsg.core as ez
from ezmsg.sigproc.base import (
    processor_state,
    BaseAdaptiveTransformer,
    BaseAdaptiveTransformerUnit,
)
from ezmsg.util.messages.axisarray import AxisArray, replace


class AdaptiveDecompSettings(ez.Settings):
    axis: str = "!time"
    n_components: int = 2


@processor_state
class AdaptiveDecompState:
    template: AxisArray | None = None
    axis_groups: tuple[str, list[str], list[str]] | None = None
    estimator: typing.Any = None


EstimatorType = typing.TypeVar(
    "EstimatorType", bound=typing.Union[IncrementalPCA, MiniBatchNMF]
)


class AdaptiveDecompTransformer(
    BaseAdaptiveTransformer[
        AdaptiveDecompSettings, AxisArray, AxisArray, AdaptiveDecompState
    ],
    typing.Generic[EstimatorType],
):
    """
    Base class for adaptive decomposition transformers. See IncrementalPCATransformer and MiniBatchNMFTransformer
    for concrete implementations.

    Note that for these classes, adaptation is not automatic. The user must call partial_fit on the transformer.
    For automated adaptation, see IncrementalDecompTransformer.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._state.estimator = self._create_estimator()

    @classmethod
    def get_message_type(cls, dir: str) -> typing.Type[AxisArray]:
        # Override because we don't reuse the generic types.
        return AxisArray

    @classmethod
    def get_estimator_type(cls) -> typing.Type[EstimatorType]:
        return typing.get_args(cls.__orig_bases__[0])[0]

    def _create_estimator(self) -> EstimatorType:
        estimator_klass = self.get_estimator_type()
        estimator_settings = self.settings.__dict__.copy()
        estimator_settings.pop("axis")
        return estimator_klass(**estimator_settings)

    def _calculate_axis_groups(self, message: AxisArray):
        if self.settings.axis.startswith("!"):
            # Iterate over the !axis and collapse all other axes
            iter_axis = self.settings.axis[1:]
            it_ax_ix = message.get_axis_idx(iter_axis)
            targ_axes = message.dims[:it_ax_ix] + message.dims[it_ax_ix + 1 :]
            off_targ_axes = []
        else:
            # Do PCA on the parameterized axis
            targ_axes = [self.settings.axis]
            # Iterate over streaming axis
            iter_axis = "win" if "win" in message.dims else "time"
            if iter_axis == self.settings.axis:
                raise ValueError(
                    f"Iterating axis ({iter_axis}) cannot be the same as the target axis ({self.settings.axis})"
                )
            it_ax_ix = message.get_axis_idx(iter_axis)
            # Remaining axes are to be treated independently
            off_targ_axes = [
                _
                for _ in (message.dims[:it_ax_ix] + message.dims[it_ax_ix + 1 :])
                if _ != self.settings.axis
            ]
        self._state.axis_groups = iter_axis, targ_axes, off_targ_axes

    def _hash_message(self, message: AxisArray) -> int:
        iter_axis = (
            self.settings.axis[1:]
            if self.settings.axis.startswith("!")
            else ("win" if "win" in message.dims else "time")
        )
        ax_idx = message.get_axis_idx(iter_axis)
        sample_shape = message.data.shape[:ax_idx] + message.data.shape[ax_idx + 1 :]
        return hash((sample_shape, message.key))

    def _reset_state(self, message: AxisArray) -> None:
        """Reset state"""
        self._calculate_axis_groups(message)
        iter_axis, targ_axes, off_targ_axes = self._state.axis_groups

        # Template
        out_dims = [iter_axis] + off_targ_axes
        out_axes = {
            iter_axis: message.axes[iter_axis],
            **{k: message.axes[k] for k in off_targ_axes},
        }
        if len(targ_axes) == 1:
            targ_ax_name = targ_axes[0]
        else:
            targ_ax_name = "components"
        out_dims += [targ_ax_name]
        out_axes[targ_ax_name] = AxisArray.CoordinateAxis(
            data=np.arange(self.settings.n_components).astype(str),
            dims=[targ_ax_name],
            unit="component",
        )
        out_shape = [message.data.shape[message.get_axis_idx(_)] for _ in off_targ_axes]
        out_shape = (0,) + tuple(out_shape) + (self.settings.n_components,)
        self._state.template = replace(
            message,
            data=np.zeros(out_shape, dtype=float),
            dims=out_dims,
            axes=out_axes,
        )

    def _process(self, message: AxisArray) -> AxisArray:
        iter_axis, targ_axes, off_targ_axes = self._state.axis_groups
        ax_idx = message.get_axis_idx(iter_axis)
        in_dat = message.data

        if in_dat.shape[ax_idx] == 0:
            return self._state.template

        # Re-order axes
        sorted_dims_exp = [iter_axis] + off_targ_axes + targ_axes
        if message.dims != sorted_dims_exp:
            # TODO: Implement axes transposition if needed
            # re_order = [ax_idx] + off_targ_inds + targ_inds
            # np.transpose(in_dat, re_order)
            pass

        # fold [iter_axis] + off_targ_axes together and fold targ_axes together
        d2 = np.prod(in_dat.shape[len(off_targ_axes) + 1 :])
        in_dat = in_dat.reshape((-1, d2))

        replace_kwargs = {
            "axes": {**self._state.template.axes, iter_axis: message.axes[iter_axis]},
        }

        # Transform data
        if hasattr(self._state.estimator, "components_"):
            decomp_dat = self._state.estimator.transform(in_dat).reshape(
                (-1,) + self._state.template.data.shape[1:]
            )
            replace_kwargs["data"] = decomp_dat

        return replace(self._state.template, **replace_kwargs)

    def partial_fit(self, message: AxisArray) -> None:
        # Check if we need to reset state
        msg_hash = self._hash_message(message)
        if self._hash != msg_hash:
            self._reset_state(message)
            self._hash = msg_hash

        iter_axis, targ_axes, off_targ_axes = self._state.axis_groups
        ax_idx = message.get_axis_idx(iter_axis)
        in_dat = message.data

        if in_dat.shape[ax_idx] == 0:
            return

        # Re-order axes if needed
        sorted_dims_exp = [iter_axis] + off_targ_axes + targ_axes
        if message.dims != sorted_dims_exp:
            # TODO: Implement axes transposition if needed
            pass

        # fold [iter_axis] + off_targ_axes together and fold targ_axes together
        d2 = np.prod(in_dat.shape[len(off_targ_axes) + 1 :])
        in_dat = in_dat.reshape((-1, d2))

        # Fit the estimator
        self._state.estimator.partial_fit(in_dat)


class IncrementalPCASettings(AdaptiveDecompSettings):
    # Additional settings specific to PCA
    whiten: bool = False
    batch_size: typing.Optional[int] = None


class IncrementalPCATransformer(AdaptiveDecompTransformer[IncrementalPCA]):
    pass


class MiniBatchNMFSettings(AdaptiveDecompSettings):
    # Additional settings specific to NMF
    init: typing.Optional[str] = "random"
    """
    'random', 'nndsvd', 'nndsvda', 'nndsvdar', 'custom', or None
    """

    batch_size: int = 1024
    """
    batch_size is used only when doing a full fit (i.e., a reset),
    or as the exponent to forget_factor, where a very small batch_size
    will cause the model to update more slowly.
    It is better to set batch_size to a larger number than the expected
    chunk size and instead use forget_factor to control the learning rate.
    """

    beta_loss: typing.Union[str, float] = "frobenius"
    """
    'frobenius', 'kullback-leibler', 'itakura-saito'
    Note that values different from 'frobenius'
        (or 2) and 'kullback-leibler' (or 1) lead to significantly slower
        fits. Note that for `beta_loss <= 0` (or 'itakura-saito'), the input
        matrix `X` cannot contain zeros.
    """

    tol: float = 1e-4

    max_no_improvement: typing.Optional[int] = None

    max_iter: int = 200

    alpha_W: float = 0.0

    alpha_H: typing.Union[float, str] = "same"

    l1_ratio: float = 0.0

    forget_factor: float = 0.7


class MiniBatchNMFTransformer(AdaptiveDecompTransformer[MiniBatchNMF]):
    pass


SettingsType = typing.TypeVar(
    "SettingsType", bound=typing.Union[IncrementalPCASettings, MiniBatchNMFSettings]
)
TransformerType = typing.TypeVar(
    "TransformerType",
    bound=typing.Union[IncrementalPCATransformer, MiniBatchNMFTransformer],
)


class BaseAdaptiveDecompUnit(
    BaseAdaptiveTransformerUnit[
        SettingsType,
        AxisArray,
        AxisArray,
        TransformerType,
    ],
    typing.Generic[SettingsType, TransformerType],
):
    INPUT_SAMPLE = ez.InputStream(AxisArray)

    @ez.subscriber(INPUT_SAMPLE)
    async def on_sample(self, msg: AxisArray) -> None:
        await self.processor.apartial_fit(msg)


class IncrementalPCAUnit(
    BaseAdaptiveDecompUnit[
        IncrementalPCASettings,
        IncrementalPCATransformer,
    ]
):
    SETTINGS = IncrementalPCASettings


class MiniBatchNMFUnit(
    BaseAdaptiveDecompUnit[
        MiniBatchNMFSettings,
        MiniBatchNMFTransformer,
    ]
):
    SETTINGS = MiniBatchNMFSettings
