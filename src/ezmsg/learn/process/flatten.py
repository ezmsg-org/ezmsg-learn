import math

import ezmsg.core as ez
import numpy as np
from array_api_compat import get_namespace
from ezmsg.baseproc import BaseTransformer, BaseTransformerUnit
from ezmsg.util.messages.axisarray import AxisArray, replace


def _normalize_axis_label(label):
    dtype_names = getattr(getattr(label, "dtype", None), "names", None)
    if dtype_names is not None:
        if "label" in dtype_names:
            return str(label["label"])
        return tuple((name, _normalize_axis_label(label[name])) for name in dtype_names)

    if isinstance(label, np.generic):
        return label.item()

    try:
        hash(label)
        return label
    except TypeError:
        return str(label)


def _axis_labels(axis_data) -> list:
    return [_normalize_axis_label(label) for label in axis_data]


class FlattenSettings(ez.Settings):
    preserve_axis: str | None = None
    sample_axis: str | None = None
    feature_axis: str = "ch"


class FlattenTransformer(BaseTransformer[FlattenSettings, AxisArray, AxisArray]):
    def _process(self, message: AxisArray) -> AxisArray:
        preserve_axis = self.settings.preserve_axis or message.dims[0]
        if preserve_axis not in message.dims:
            raise ValueError(f"Axis {preserve_axis} not found in message dims {message.dims}.")

        sample_axis = self.settings.sample_axis or preserve_axis
        feature_axis = self.settings.feature_axis
        if sample_axis == feature_axis:
            raise ValueError("sample_axis and feature_axis must be different.")

        preserve_idx = message.get_axis_idx(preserve_axis)
        xp = get_namespace(message.data)

        if preserve_idx == 0:
            data = message.data
        else:
            perm = (preserve_idx,) + tuple(idx for idx in range(message.data.ndim) if idx != preserve_idx)
            data = xp.permute_dims(message.data, perm)

        n_features = math.prod(data.shape[1:]) if data.ndim > 1 else 1
        flat = xp.reshape(data, (data.shape[0], n_features))

        sample_ax = message.axes.get(preserve_axis)
        if sample_ax is not None and hasattr(sample_ax, "dims"):
            sample_ax = replace(sample_ax, dims=[sample_axis])

        flat_dims = [dim for dim in message.dims if dim != preserve_axis]
        if (
            len(flat_dims) == 1
            and flat_dims[0] == feature_axis
            and feature_axis in message.axes
            and hasattr(message.axes[feature_axis], "data")
        ):
            feature_ax = message.axes[feature_axis]
            if hasattr(feature_ax, "dims"):
                feature_ax = replace(feature_ax, dims=[feature_axis])
        elif (
            flat_dims == [sample_axis, feature_axis]
            and data.ndim == 3
            and feature_axis in message.axes
            and hasattr(message.axes[feature_axis], "data")
        ):
            base_labels = [str(label) for label in _axis_labels(message.axes[feature_axis].data)]
            if len(base_labels) != data.shape[2]:
                raise ValueError(f"Expected {data.shape[2]} feature labels, got {len(base_labels)}.")
            expanded_labels = [f"{label}__t-{lag}" for lag in range(data.shape[1] - 1, -1, -1) for label in base_labels]
            feature_ax = AxisArray.CoordinateAxis(
                data=np.asarray(expanded_labels),
                dims=[feature_axis],
            )
        else:
            feature_ax = AxisArray.CoordinateAxis(
                data=np.arange(n_features),
                dims=[feature_axis],
            )

        axes = {feature_axis: feature_ax}
        if sample_ax is not None:
            axes[sample_axis] = sample_ax

        return replace(
            message,
            data=flat,
            dims=[sample_axis, feature_axis],
            axes=axes,
        )


class Flatten(BaseTransformerUnit[FlattenSettings, AxisArray, AxisArray, FlattenTransformer]):
    SETTINGS = FlattenSettings
