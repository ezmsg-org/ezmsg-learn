"""
TODO: This needs a lot of work and some testig.
This node will assume that all incoming data will be aligned on the same time basis.
INPUT_TRIGGER = ez.InputStream(SampleTriggerMessage)
The .value is a mostly useless string. Reserved for special commands like "stop"
We may need task-specific TriggerParser or similar node to interpret task events
and convert them to SampleTriggerMessage.
INPUT_VALUE = ez.InputStream(AxisArray) takes in the continuous signal to be used as the labels and buffers them.
INPUT_SIGNAL = ez.InputStream(AxisArray) takes in the continuous data and buffers them.
OUTPUT_SAMPLE = ez.OutputStream(AxisArray)

max_buffer_size: int = 512*1024*1024 to put a cap on memory usage for each of _value_buffer and _signal_buffer

"""

import asyncio
import copy

import ezmsg.core as ez
import numpy as np
from ezmsg.sigproc.sampler import SampleTriggerMessage
from ezmsg.util.messages.axisarray import AxisArray, slice_along_axis
from ezmsg.util.messages.util import replace


class SeqSeqSampler:
    def __init__(
        self,
        max_buffer_dur: float = 5.0,
    ):
        self._max_buffer_dur = max_buffer_dur
        self._trig_queue: asyncio.Queue[SampleTriggerMessage] = asyncio.Queue()
        self._value_buffer: AxisArray | None = None
        self._signal_buffer: AxisArray | None = None

    def __aiter__(self):
        self._trig_queue: asyncio.Queue[SampleTriggerMessage] = asyncio.Queue()
        self._value_buffer: AxisArray | None = None
        self._signal_buffer: AxisArray | None = None
        return self

    async def asend(self, message: AxisArray):
        await self.enqueue_signal(message)
        return await self.__anext__()

    async def enqueue_signal(self, message: AxisArray):
        self._update_buffer(message, "signal")

    async def enqueue_value(self, message: AxisArray):
        self._update_buffer(message, "value")

    async def enqueue_trigger(self, message: SampleTriggerMessage):
        if isinstance(message.value, str) and message.value == "end":
            # TODO: For each trigger currently in self._trig_queue, overwrite its
            #  `.period[1]` with the incoming trigger's .timestamp
            print("TODO")
        else:
            await self._trig_queue.put(message)

    async def __anext__(self):
        try:
            trig = self._trig_queue.get_nowait()
        except asyncio.QueueEmpty:
            return None
        samp_msg = self._process_trigger(trig)

        if samp_msg is None:
            # Trigger could not be processed fully because buffers did not satisfy the period.
            await self._trig_queue.put(trig)

        self._trig_queue.task_done()
        return samp_msg

    def _process_trigger(self, trig: SampleTriggerMessage) -> AxisArray | None:
        trig_range = trig.timestamp + np.array(trig.period)
        if self._value_buffer is None or self._signal_buffer is None:
            return None
        if hasattr(self._value_buffer.axes["time"], "data"):
            val_tvec = self._value_buffer.axes["time"].data
        else:
            n_vals = self._value_buffer.data.shape[self._value_buffer.get_axis_idx("time")]
            val_tvec = self._value_buffer.axes["time"].value(np.arange(n_vals))
        if hasattr(self._signal_buffer.axes["time"], "data"):
            sig_tvec = self._signal_buffer.axes["time"].data
        else:
            n_times = self._signal_buffer.data.shape[self._signal_buffer.get_axis_idx("time")]
            sig_tvec = self._signal_buffer.axes["time"].value(np.arange(n_times))
        b_filled = val_tvec.size > 0 and sig_tvec.size > 0
        b_filled = b_filled and val_tvec[-1] >= trig_range[1] and sig_tvec[-1] >= trig_range[1]
        if b_filled:
            messages: dict[str, AxisArray] = {}
            for buf_name, buffer, tvec in [
                ("value", self._value_buffer, val_tvec),
                ("signal", self._signal_buffer, sig_tvec),
            ]:
                keep_inds = np.where(np.logical_and(tvec >= trig_range[0], tvec < trig_range[1]))[0]
                if hasattr(buffer.axes["time"], "data"):
                    new_time_ax = replace(buffer.axes["time"], data=tvec[keep_inds])
                else:
                    new_time_ax = replace(buffer.axes["time"], offset=tvec[keep_inds[0]])
                new_dat = slice_along_axis(
                    buffer.data,
                    slice(keep_inds[0], keep_inds[-1] + 1),
                    axis=buffer.get_axis_idx("time"),
                )
                new_msg = replace(
                    buffer,
                    data=new_dat,
                    axes={
                        **buffer.axes,
                        "time": new_time_ax,
                    },
                )
                messages[buf_name] = new_msg

            trig.value = messages["value"]
            samp_msg = messages["signal"]
            samp_msg.attrs["trigger"] = trig
        else:
            samp_msg = None

        return samp_msg

    def _update_buffer(self, message: AxisArray, target: str):
        if target == "value":
            buffer = self._value_buffer
        elif target == "signal":
            buffer = self._signal_buffer
        else:
            raise ValueError(f"Invalid target: {target}")

        ax_ix = message.get_axis_idx("time")
        # TODO: Check if we need to reset the buffer because the input changed.

        if buffer is None:
            buffer = copy.deepcopy(message)
            if target == "value":
                self._value_buffer = buffer
            elif target == "signal":
                self._signal_buffer = buffer
        else:
            buffer.data = np.concatenate([buffer.data, message.data], axis=ax_ix)
            if hasattr(buffer.axes["time"], "data"):
                buffer.axes["time"].data = np.concatenate([buffer.axes["time"].data, message.axes["time"].data], axis=0)
            # No need for `else:` condition because offset does not change.

        # Trim down to self._max_buffer_dur
        if hasattr(buffer.axes["time"], "data"):
            tvec = buffer.axes["time"].data
        else:
            n_times = buffer.data.shape[buffer.get_axis_idx("time")]
            tvec = buffer.axes["time"].value(np.arange(n_times))
        t_min = tvec[-1] - self._max_buffer_dur
        b_keep = tvec >= t_min
        if not np.all(b_keep):
            keep_inds = np.where(b_keep)[0]
            buffer.data = slice_along_axis(buffer.data, slice(keep_inds[0], keep_inds[-1] + 1), ax_ix)
            tvec = tvec[keep_inds]
            if hasattr(buffer.axes["time"], "data"):
                buffer.axes["time"].data = tvec
            else:
                buffer.axes["time"].offset = tvec[0]


class SeqSeqSamplerSettings(ez.Settings):
    max_buffer_dur: float = 5.0


class SeqSeqSamplerState(ez.State):
    core: SeqSeqSampler


class SeqSeqSamplerUnit(ez.Unit):
    SETTINGS = SeqSeqSamplerSettings
    STATE = SeqSeqSamplerState

    INPUT_TRIGGER = ez.InputStream(SampleTriggerMessage)
    INPUT_VALUE = ez.InputStream(AxisArray)
    INPUT_SIGNAL = ez.InputStream(AxisArray)
    OUTPUT_SAMPLE = ez.OutputStream(AxisArray)

    async def initialize(self):
        self.STATE.core = SeqSeqSampler(max_buffer_dur=self.SETTINGS.max_buffer_dur)

    @ez.subscriber(INPUT_TRIGGER, zero_copy=True)
    async def on_trigger(self, message: SampleTriggerMessage):
        await self.STATE.core.enqueue_trigger(message)

    @ez.subscriber(INPUT_VALUE, zero_copy=True)
    async def on_value(self, message: AxisArray):
        await self.STATE.core.enqueue_value(message)

    @ez.subscriber(INPUT_SIGNAL, zero_copy=True)
    async def on_signal(self, message: AxisArray):
        await self.STATE.core.enqueue_signal(message)

    @ez.publisher(OUTPUT_SAMPLE)
    async def send_sample(self):
        while True:
            result: AxisArray = await anext(self.STATE.core)
            if result is not None:
                yield self.OUTPUT_SAMPLE, result
            else:
                # No sample could be produced. Try again later.
                await asyncio.sleep(0.005)
