import typing

import numpy as np
from sklearn.decomposition import IncrementalPCA, MiniBatchNMF
import ezmsg.core as ez
from ezmsg.util.messages.axisarray import AxisArray
from ezmsg.util.messages.util import replace
from ezmsg.sigproc.base import BaseSignalTransformer, BaseSignalTransformerUnit
from ezmsg.sigproc.window import windowing


class IncrementalDecompSettings(ez.Settings):
    axis: str = "!time"
    n_components: int = 2
    update_interval: float = 0.0
    method: str = "pca"


class IncrementalDecompState(ez.State):
    wingen: typing.Iterator | None = None  # TODO: wingen.state, not wingen itself
    decomp: IncrementalPCA | MiniBatchNMF | None = None  # TODO: decomp.state, not decomp itself
    hash: int = 0


class IncrementalDecompTransformer(
    BaseSignalTransformer[IncrementalDecompState, IncrementalDecompSettings, AxisArray]
):
    def _get_axis_groups(self, message: AxisArray) -> tuple[str, list[str], list[str]]:
        if self.settings.axis.startswith("!"):
            iter_axis = self.settings.axis[1:]
            it_ax_ix = message.get_axis_idx(iter_axis)
            targ_axes = message.dims[:it_ax_ix] + message.dims[it_ax_ix + 1:]
            off_targ_axes = []
        else:
            targ_axes = [self.settings.axis]
            iter_axis = "win" if "win" in message.dims else "time"
            it_ax_ix = message.get_axis_idx(iter_axis)
            off_targ_axes = [_ for _ in (message.dims[:it_ax_ix] + message.dims[it_ax_ix + 1:]) if _ != self.settings.axis]
        return iter_axis, targ_axes, off_targ_axes

    def check_metadata(self, message: AxisArray) -> bool:
        iter_axis = self.settings.axis[1:] if self.settings.axis.startswith("!") else ("win" if "win" in message.dims else "time")
        ax_idx = message.get_axis_idx(iter_axis)
        sample_shape = message.data.shape[:ax_idx] + message.data.shape[ax_idx + 1:]
        in_hash = hash((sample_shape, message.key))
        return in_hash != self.state.hash

    def reset(self, message: AxisArray) -> None:
        # Reset state

        # Hash
        iter_axis, targ_axes, off_targ_axes = self._get_axis_groups(message)
        ax_idx = message.get_axis_idx(iter_axis)
        sample_shape = message.data.shape[:ax_idx] + message.data.shape[ax_idx + 1:]
        self.state.hash = hash((sample_shape, message.key))

        # Decomposition object
        if self.settings.method == "pca":
            self.state.decomp = IncrementalPCA(n_components=self.settings.n_components)
        elif self.settings.method == "nmf":
            self.state.decomp = MiniBatchNMF(n_components=self.settings.n_components)
        else:
            raise ValueError(f"Unknown method: {self.settings.method}")

        # Windower
        if self.settings.update_interval != 0:
            self.state.wingen = windowing(
                axis=iter_axis,
                window_dur=self.settings.update_interval,
                window_shift=self.settings.update_interval,
                zero_pad_until="none",  # zero-padding would throw off PCA
            )
        else:
            self.state.wingen = None

        # Template
        out_dims = [iter_axis] + off_targ_axes
        out_axes = {iter_axis: message.axes[iter_axis], **{k: message.axes[k] for k in off_targ_axes}}
        if len(targ_axes) == 1:
            targ_ax_name = targ_axes[0]
        else:
            targ_ax_name = "components"
        out_dims += [targ_ax_name]
        out_axes[targ_ax_name] = AxisArray.CoordinateAxis(
            data=np.arange(self.settings.n_components).astype(str), dims=[targ_ax_name], unit="component"
        )
        out_shape = [message.data.shape[message.get_axis_idx(_)] for _ in off_targ_axes]
        out_shape = (0,) + tuple(out_shape) + (self.settings.n_components,)
        self.state.template = AxisArray(
            data=np.zeros(out_shape, dtype=float),
            dims=out_dims,
            axes=out_axes,
            key=message.key,
        )

    def _process(self, message: AxisArray) -> AxisArray:
        iter_axis, targ_axes, off_targ_axes = self._get_axis_groups(message)
        ax_idx = message.get_axis_idx(iter_axis)
        in_dat = message.data

        if in_dat.shape[ax_idx] == 0:
            return self.state.template

        # Re-order axes
        sorted_dims_exp = [iter_axis] + off_targ_axes + targ_axes
        if message.dims != sorted_dims_exp:
            print("TODO: Transpose in_dat from message.dims -> [iter_axis] + off_targ_axes + targ_axes")
            # re_order = [ax_idx] + off_targ_inds + targ_inds
            # np.transpose(in_dat, re_order)

        # fold [iter_axis] + off_targ_axes together and fold targ_axes together
        d2 = np.prod(in_dat.shape[len(off_targ_axes) + 1:])
        in_dat = in_dat.reshape((-1, d2))

        # Prepare training data
        if self.state.wingen is None or not hasattr(self.state.decomp, "components_"):
            # No windowing or this is the first pass
            self.state.decomp.partial_fit(in_dat)
        else:
            train_msg = self.state.wingen.send(message)
            _shp = train_msg.data.shape
            new_shape = (
                    _shp[:ax_idx]
                    + (np.prod(_shp[ax_idx: ax_idx + 2]),)
                    + _shp[ax_idx + 2:]
            )
            train_dat = train_msg.data.reshape(new_shape)
            if message.dims != sorted_dims_exp:
                print("TODO: Transpose train_dat from message.dims -> [iter_axis] + off_targ_axes + targ_axes")
                # np.transpose(train_dat, re_order)
            if np.prod(train_dat.shape):
                train_dat = train_dat.reshape((-1, d2))
                self.state.decomp.partial_fit(train_dat)

        decomp_dat = self.state.decomp.transform(in_dat).reshape((-1,) + self.state.template.data.shape[1:])
        return replace(self.state.template, data=decomp_dat)


class IncrementalDecomp(
    BaseSignalTransformerUnit[IncrementalDecompState, IncrementalDecompSettings, AxisArray, IncrementalDecompTransformer]
):
    SETTINGS = IncrementalDecompSettings
