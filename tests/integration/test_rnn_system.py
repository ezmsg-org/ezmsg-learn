import os
import tempfile
from pathlib import Path

import ezmsg.core as ez
from ezmsg.sigproc.synth import Counter, CounterSettings
from ezmsg.util.messagecodec import message_log
from ezmsg.util.messagelogger import MessageLogger, MessageLoggerSettings
from ezmsg.util.messages.axisarray import AxisArray
from ezmsg.util.terminate import TerminateOnTotal, TerminateOnTotalSettings

from ezmsg.learn.process.rnn import RNNUnit


def test_torch_model_unit_system():
    fs = 10.0
    block_size = 4
    duration = 2.0  # seconds
    input_size = 3
    output_size = 2
    hidden_size = 30
    num_layers = 2
    single_precision = True

    test_filename = Path(tempfile.gettempdir())
    test_filename = test_filename / Path("test_torch_system.txt")
    with open(test_filename, "w"):
        pass
    ez.logger.info(f"Logging to {test_filename}")

    comps = {
        "SRC": Counter(
            CounterSettings(
                fs=fs,
                n_ch=input_size,
                n_time=block_size,
                dispatch_rate=duration,
                mod=None,
            )
        ),
        "MODEL": RNNUnit(
            single_precision=single_precision,
            learning_rate=1e-2,
            scheduler_gamma=0.0,
            weight_decay=0.0,
            device="cpu",
            model_kwargs={
                "hidden_size": hidden_size,
                "num_layers": num_layers,
                "output_size": output_size,
            },
        ),
        "LOG": MessageLogger(MessageLoggerSettings(output=test_filename)),
        "TERM": TerminateOnTotal(
            TerminateOnTotalSettings(total=int(duration * fs / block_size))
        ),
    }

    conns = (
        (comps["SRC"].OUTPUT_SIGNAL, comps["MODEL"].INPUT_SIGNAL),
        (comps["MODEL"].OUTPUT_SIGNAL, comps["LOG"].INPUT_MESSAGE),
        (comps["LOG"].OUTPUT_MESSAGE, comps["TERM"].INPUT_MESSAGE),
    )

    ez.run(components=comps, connections=conns)

    # Read from message log
    messages: list[AxisArray] = [_ for _ in message_log(test_filename)]
    os.remove(test_filename)

    # Check basic structure
    assert all(msg.data.shape[-1] == output_size for msg in messages)
    assert all("time" in msg.dims and "ch" in msg.dims for msg in messages)
    assert all("ch" in msg.axes for msg in messages)
    assert messages[0].axes["ch"].data.shape[0] == output_size
