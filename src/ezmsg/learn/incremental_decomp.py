import pickle

import numpy as np
from sklearn.decomposition import IncrementalPCA, MiniBatchNMF
import ezmsg.core as ez
from ezmsg.util.messages.axisarray import AxisArray
from ezmsg.util.messages.util import replace
from ezmsg.sigproc.base import (
    CompositeProcessor,
    BaseTransformerUnit,
    ProcessorState,
    BaseStatefulProcessor,
)
from ezmsg.sigproc.window import WindowTransformer


class IncrementalDecompSettings(ez.Settings):
    axis: str = "!time"
    n_components: int = 2
    update_interval: float = 0.0
    method: str = "pca"
    # TODO: More parameters needed, especially for NMF


class IncrementalDecompState(ProcessorState):
    template: AxisArray | None = None
    axis_groups: tuple[str, list[str], list[str]] | None = None


class IncrementalDecompTransformer(
    CompositeProcessor[IncrementalDecompSettings, AxisArray]
):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._state: IncrementalDecompState = IncrementalDecompState()

    @staticmethod
    def _initialize_processors(
        settings: IncrementalDecompSettings,
    ) -> dict[str, BaseStatefulProcessor]:
        return {
            "decomp": IncrementalPCA(n_components=settings.n_components)
            if settings.method == "pca"
            else MiniBatchNMF(n_components=settings.n_components),
            # Create temporary windowing processor. It is temporary because the `axis` is just a guess.
            "windowing": WindowTransformer(
                axis="time",
                window_dur=settings.update_interval,
                window_shift=settings.update_interval,
                zero_pad_until="none",
            )
            if settings.update_interval != 0
            else None,
        }

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

        # Reset windowing with correct iter_axis
        if self.settings.update_interval != 0:
            # TODO: Verify update_interval corresponds to at least as many samples as
            #  the product of the remaining shape.
            self._procs["windowing"] = WindowTransformer(
                axis=iter_axis,
                window_dur=self.settings.update_interval,
                window_shift=self.settings.update_interval,
                zero_pad_until="none",
            )

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
        self._state.template = AxisArray(
            data=np.zeros(out_shape, dtype=float),
            dims=out_dims,
            axes=out_axes,
            key=message.key,
        )

    @property
    def state(self) -> dict[str, ProcessorState]:
        return {**super().state, "self": self._state}

    @state.setter
    def state(
        self,
        state: dict[str, ProcessorState | IncrementalDecompSettings] | bytes | None,
    ) -> None:
        if state is not None:
            if isinstance(state, bytes):
                state = pickle.loads(state)
            self._state = state.pop("self")
            super().state = state

    def __call__(self, message: AxisArray) -> AxisArray:
        # Override __call__ with the one from BaseStatefulProcessor
        msg_hash = self._hash_message(message)
        if msg_hash != self._state.hash:
            self._reset_state(message)
            self._state.hash = msg_hash
        return self._process(message)

    def _process(self, message: AxisArray) -> AxisArray:
        """
        Do not call this directly! It should be called by __call__.
        Override super's _process because we must first reshape the message before passing it off to the sub-processors.

        :param message: Input message
        :return: Decomposed version of input message.

        """
        iter_axis, targ_axes, off_targ_axes = self._state.axis_groups
        ax_idx = message.get_axis_idx(iter_axis)
        in_dat = message.data

        if in_dat.shape[ax_idx] == 0:
            return self._state.template

        # Re-order axes
        sorted_dims_exp = [iter_axis] + off_targ_axes + targ_axes
        if message.dims != sorted_dims_exp:
            print(
                "TODO: Transpose in_dat from message.dims -> [iter_axis] + off_targ_axes + targ_axes"
            )
            # re_order = [ax_idx] + off_targ_inds + targ_inds
            # np.transpose(in_dat, re_order)

        # fold [iter_axis] + off_targ_axes together and fold targ_axes together
        d2 = np.prod(in_dat.shape[len(off_targ_axes) + 1 :])
        in_dat = in_dat.reshape((-1, d2))

        # Prepare training data
        if self._procs["windowing"] is None or not hasattr(
            self._procs["decomp"], "components_"
        ):
            # No windowing or this is the first pass
            self._procs["decomp"].partial_fit(in_dat)
        else:
            train_msg = self._procs["windowing"].send(message)
            _shp = train_msg.data.shape
            new_shape = (
                _shp[:ax_idx]
                + (np.prod(_shp[ax_idx : ax_idx + 2]),)
                + _shp[ax_idx + 2 :]
            )
            train_dat = train_msg.data.reshape(new_shape)
            if message.dims != sorted_dims_exp:
                print(
                    "TODO: Transpose train_dat from message.dims -> [iter_axis] + off_targ_axes + targ_axes"
                )
                # np.transpose(train_dat, re_order)
            if np.prod(train_dat.shape):
                train_dat = train_dat.reshape((-1, d2))
                self._procs["decomp"].partial_fit(train_dat)

        decomp_dat = (
            self._procs["decomp"]
            .transform(in_dat)
            .reshape((-1,) + self._state.template.data.shape[1:])
        )
        return replace(self._state.template, data=decomp_dat)


class IncrementalDecomp(
    BaseTransformerUnit[
        IncrementalDecompSettings, AxisArray, IncrementalDecompTransformer
    ]
):
    SETTINGS = IncrementalDecompSettings
