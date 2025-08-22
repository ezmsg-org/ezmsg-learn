import pytest

import numpy as np
from ezmsg.util.messages.axisarray import AxisArray, replace
from ezmsg.sigproc.sampler import SampleTriggerMessage, SampleMessage

from ezmsg.learn.process.adaptive_linear_regressor import (
    AdaptiveLinearRegressorTransformer,
)


@pytest.mark.parametrize("model_type", ["linear", "logistic", "sgd", "par"])
def test_adaptive_linear_regressor(model_type: str):
    n_ch = 3
    dur = 1.0
    fs = 1000.0
    n_times = int(dur * fs)

    model_coefs = np.arange(1, n_ch + 1)
    X = np.arange(np.prod((n_times, n_ch))).reshape((n_times, n_ch)).astype(float)
    y = (np.dot(X, model_coefs) + (n_ch + 1))[..., None]

    value_axarr = AxisArray(
        data=y,
        dims=["time", "ch"],
        axes={
            "time": AxisArray.TimeAxis(fs=fs, offset=0),
            "ch": AxisArray.CoordinateAxis(data=np.array(["y"]), dims=["ch"]),
        },
        key="value",
    )
    sig_axarr = AxisArray(
        data=X + np.random.randn(*X.shape),
        dims=["time", "ch"],
        axes={
            "time": AxisArray.TimeAxis(fs=fs, offset=0),
            "ch": AxisArray.CoordinateAxis(
                data=np.array([f"X{_}" for _ in range(n_ch)]), dims=["ch"]
            ),
        },
        key="signal",
    )
    samp_trig = SampleTriggerMessage(
        timestamp=0.0,
        period=(0.0, dur),
        value=value_axarr,
    )
    samp = SampleMessage(trigger=samp_trig, sample=sig_axarr)

    proc = AdaptiveLinearRegressorTransformer(model_type=model_type)
    _ = proc.send(samp)
    preds = proc.send(replace(sig_axarr, data=X + np.random.randn(*X.shape)))
    assert isinstance(preds, AxisArray)
    assert preds.data.shape == (n_times, 1)
