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

MAX_ONE_TO_ONE_SAMPLE_MISMATCH = 1


def _time_vector(message: AxisArray) -> np.ndarray:
    time_axis = message.axes["time"]
    time_idx = message.get_axis_idx("time")
    n_times = message.data.shape[time_idx]
    if hasattr(time_axis, "data"):
        return np.asarray(time_axis.data)
    return np.asarray(time_axis.value(np.arange(n_times)))


def _sample_spacing(tvec: np.ndarray) -> float | None:
    if len(tvec) < 2:
        return None

    dt = np.abs(np.diff(tvec))
    dt = dt[dt > 0]
    if dt.size == 0:
        return None
    return float(np.median(dt))


def _boundary_slack(*tvecs: np.ndarray) -> float:
    spacings = [_sample_spacing(tvec) for tvec in tvecs]
    spacings = [spacing for spacing in spacings if spacing is not None]
    if not spacings:
        return 0.0
    return max(spacings)


def _select_best_aligned_window(
    longer_inds: np.ndarray,
    longer_tvec: np.ndarray,
    shorter_tvec: np.ndarray,
) -> tuple[np.ndarray, int, float]:
    """Crop a longer contiguous slice to the subwindow best aligned to shorter_tvec."""
    target_len = len(shorter_tvec)
    extra = len(longer_inds) - target_len

    if target_len <= 0:
        return longer_inds[:0], 0, 0.0

    best_offset = 0
    best_cost = float("inf")
    for offset in range(extra + 1):
        candidate_tvec = longer_tvec[offset : offset + target_len]
        cost = float(np.mean(np.abs(candidate_tvec - shorter_tvec)))
        if cost < best_cost:
            best_offset = offset
            best_cost = cost

    aligned_inds = longer_inds[best_offset : best_offset + target_len]
    return aligned_inds, best_offset, best_cost


def _align_keep_indices(
    signal_inds: np.ndarray,
    signal_tvec: np.ndarray,
    value_inds: np.ndarray,
    value_tvec: np.ndarray,
    trig: SampleTriggerMessage,
) -> tuple[np.ndarray, np.ndarray] | None:
    length_delta = abs(len(signal_inds) - len(value_inds))

    if length_delta == 0:
        return signal_inds, value_inds

    if len(signal_inds) == 0 or len(value_inds) == 0:
        return None

    if length_delta > MAX_ONE_TO_ONE_SAMPLE_MISMATCH:
        ez.logger.warning(
            "SeqSeqSampler could not align mismatched trigger window: "
            f"signal_len={len(signal_inds)} value_len={len(value_inds)} "
            f"trigger_timestamp={trig.timestamp} trigger_period={trig.period}"
        )
        return None

    if len(signal_inds) > len(value_inds):
        aligned_signal_inds, offset, cost = _select_best_aligned_window(signal_inds, signal_tvec, value_tvec)
        ez.logger.warning(
            "SeqSeqSampler aligned mismatched trigger window by trimming signal: "
            f"signal_len={len(signal_inds)} value_len={len(value_inds)} "
            f"offset={offset} mean_abs_dt={cost:.6f}s "
            f"trigger_timestamp={trig.timestamp} trigger_period={trig.period}"
        )
        return aligned_signal_inds, value_inds

    aligned_value_inds, offset, cost = _select_best_aligned_window(value_inds, value_tvec, signal_tvec)
    ez.logger.warning(
        "SeqSeqSampler aligned mismatched trigger window by trimming value: "
        f"signal_len={len(signal_inds)} value_len={len(value_inds)} "
        f"offset={offset} mean_abs_dt={cost:.6f}s "
        f"trigger_timestamp={trig.timestamp} trigger_period={trig.period}"
    )
    return signal_inds, aligned_value_inds


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
        samp_msg, keep_waiting = self._process_trigger(trig)

        if keep_waiting:
            # Trigger could not be processed fully because buffers did not satisfy the period.
            await self._trig_queue.put(trig)

        self._trig_queue.task_done()
        return samp_msg

    def _process_trigger(self, trig: SampleTriggerMessage) -> tuple[AxisArray | None, bool]:
        if trig.period is None:
            ez.logger.warning("SeqSeqSampler dropped trigger without a period.")
            return None, False

        trig_range = trig.timestamp + np.array(trig.period)
        if self._value_buffer is None or self._signal_buffer is None:
            return None, True
        val_tvec = _time_vector(self._value_buffer)
        sig_tvec = _time_vector(self._signal_buffer)
        if val_tvec.size == 0 or sig_tvec.size == 0:
            return None, True

        boundary_slack = _boundary_slack(val_tvec, sig_tvec)
        if trig_range[0] < (val_tvec[0] - boundary_slack) or trig_range[0] < (sig_tvec[0] - boundary_slack):
            ez.logger.warning(
                "SeqSeqSampler dropped trigger before buffers could satisfy it: "
                f"signal_span=({sig_tvec[0]}, {sig_tvec[-1]}) "
                f"value_span=({val_tvec[0]}, {val_tvec[-1]}) "
                f"trigger_timestamp={trig.timestamp} trigger_period={trig.period}"
            )
            return None, False

        if trig_range[1] > val_tvec[-1] or trig_range[1] > sig_tvec[-1]:
            return None, True

        value_keep_inds = np.where(np.logical_and(val_tvec >= trig_range[0], val_tvec < trig_range[1]))[0]
        signal_keep_inds = np.where(np.logical_and(sig_tvec >= trig_range[0], sig_tvec < trig_range[1]))[0]

        if len(value_keep_inds) == 0 or len(signal_keep_inds) == 0:
            ez.logger.warning(
                "SeqSeqSampler could not slice trigger window: "
                f"signal_len={len(signal_keep_inds)} value_len={len(value_keep_inds)} "
                f"trigger_timestamp={trig.timestamp} trigger_period={trig.period}"
            )
            return None, False

        aligned_keep_inds = _align_keep_indices(
            signal_keep_inds,
            sig_tvec[signal_keep_inds],
            value_keep_inds,
            val_tvec[value_keep_inds],
            trig,
        )
        if aligned_keep_inds is None:
            return None, False

        signal_keep_inds, value_keep_inds = aligned_keep_inds
        messages: dict[str, AxisArray] = {}
        for buf_name, buffer, tvec, keep_inds in [
            ("value", self._value_buffer, val_tvec, value_keep_inds),
            ("signal", self._signal_buffer, sig_tvec, signal_keep_inds),
        ]:
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

        sample_trigger = copy.copy(trig)
        sample_trigger.value = messages["value"]
        samp_msg = replace(
            messages["signal"],
            attrs={**messages["signal"].attrs, "trigger": sample_trigger},
        )
        return samp_msg, False

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

    @ez.subscriber(INPUT_TRIGGER)
    async def on_trigger(self, message: SampleTriggerMessage):
        await self.STATE.core.enqueue_trigger(message)

    @ez.subscriber(INPUT_VALUE)
    async def on_value(self, message: AxisArray):
        await self.STATE.core.enqueue_value(message)

    @ez.subscriber(INPUT_SIGNAL)
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
