import pytest

import numpy as np
from ezmsg.util.messages.axisarray import AxisArray, replace
from ezmsg.sigproc.sampler import SampleTriggerMessage, SampleMessage

from ezmsg.learn.process.linear_regressor import LinearRegressorTransformer


@pytest.mark.parametrize("model_type", ["ridge", "linear"])
def test_linear_regressor(model_type: str):
    n_ch = 3
    dur = 1.0
    fs = 1000.0
    n_times = int(dur * fs)

    model_coefs = np.arange(n_ch * 2).reshape((n_ch, 2)) + 1
    X = np.arange(np.prod((n_times, n_ch))).reshape((n_times, n_ch)).astype(float)
    y = np.dot(X, model_coefs) + (n_ch + 1)

    value_axarr = AxisArray(
        data=y,
        dims=["time", "ch"],
        axes={
            "time": AxisArray.TimeAxis(fs=fs, offset=0),
            "ch": AxisArray.CoordinateAxis(data=np.array(["x", "y"]), dims=["ch"]),
        },
        key="value",
    )
    sig_axarr = AxisArray(
        data=X + 0.1 * np.random.randn(*X.shape),
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

    gen = LinearRegressorTransformer(model_type=model_type)
    _ = gen.send(samp)
    preds = gen.send(replace(sig_axarr, data=X + 0.1 * np.random.randn(*X.shape)))
    rss = ((samp_trig.value.data - preds.data) ** 2).sum()
    tss = ((samp_trig.value.data - samp_trig.value.data.mean()) ** 2).sum()
    rsq = 1 - rss / tss
    assert rsq > 0.99
