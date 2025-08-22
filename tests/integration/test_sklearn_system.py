import os
import pickle
import tempfile
from pathlib import Path

import ezmsg.core as ez
import numpy as np
import pandas as pd
from ezmsg.sigproc.synth import Counter
from ezmsg.util.messagecodec import message_log
from ezmsg.util.messagelogger import MessageLogger
from ezmsg.util.messages.axisarray import AxisArray
from ezmsg.util.terminate import TerminateOnTotal
from river.linear_model import LinearRegression

from ezmsg.learn.process.sklearn import SklearnModelUnit


def test_sklearn_model_unit_system():
    fs = 10.0
    block_size = 4
    duration = 2.0  # seconds
    input_size = 3
    output_size = 1  # For most sklearn regressors, output is single dim

    # Create temporary checkpoint file path
    checkpoint_path = Path(tempfile.gettempdir()) / "sklearn_checkpoint.pkl"

    # Fit model and save checkpoint
    model = LinearRegression()
    X = pd.DataFrame(
        np.random.randn(block_size, input_size),
        columns=[f"f{i}" for i in range(input_size)],
    )
    y = pd.Series(np.random.randn(block_size), name="target")
    model.learn_many(X, y)
    with open(checkpoint_path, "wb") as f:
        pickle.dump(model, f)

    test_filename = Path(tempfile.gettempdir()) / "test_sklearn_system.txt"
    with open(test_filename, "w"):
        pass
    ez.logger.info(f"Logging to {test_filename}")

    comps = {
        "SRC": Counter(
            fs=fs,
            n_ch=input_size,
            n_time=block_size,
            dispatch_rate=duration,
            mod=None,
        ),
        "MODEL": SklearnModelUnit(
            model_class="river.linear_model.LinearRegression",
            model_kwargs={},
            checkpoint_path=str(checkpoint_path),
            partial_fit_classes=None,
        ),
        "LOG": MessageLogger(output=test_filename),
        "TERM": TerminateOnTotal(total=int(duration * fs / block_size)),
    }

    conns = (
        (comps["SRC"].OUTPUT_SIGNAL, comps["MODEL"].INPUT_SIGNAL),
        (comps["MODEL"].OUTPUT_SIGNAL, comps["LOG"].INPUT_MESSAGE),
        (comps["LOG"].OUTPUT_MESSAGE, comps["TERM"].INPUT_MESSAGE),
    )

    # Run the pipeline
    ez.run(components=comps, connections=conns)

    # Read logged messages
    messages: list[AxisArray] = list(message_log(test_filename))

    # Clean up the temporary files
    os.remove(test_filename)
    os.remove(checkpoint_path)

    # Assertions
    # Output shape should have same number of samples, channels = output_size
    assert all(msg.data.shape[0] == block_size for msg in messages)
    assert all(msg.data.shape[1] == output_size for msg in messages)

    # Dimensions and axes presence checks
    assert all("time" in msg.dims and "ch" in msg.dims for msg in messages)
    assert all("ch" in msg.axes for msg in messages)
    assert messages[0].axes["ch"].data.shape[0] == output_size
