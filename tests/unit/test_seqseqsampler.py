import asyncio

import numpy as np
from ezmsg.baseproc import SampleTriggerMessage
from ezmsg.util.messages.axisarray import AxisArray

from ezmsg.learn.process.seqseqsampler import SeqSeqSampler


def _make_signal(n_times: int, fs: float, offset: float = 0.0, start_value: float = 0.0) -> AxisArray:
    data = (start_value + np.arange(n_times, dtype=float))[:, None]
    return AxisArray(
        data=data,
        dims=["time", "ch"],
        axes={
            "time": AxisArray.TimeAxis(fs=fs, offset=offset),
            "ch": AxisArray.CoordinateAxis(data=np.array(["signal"]), dims=["ch"]),
        },
        key="signal",
    )


def _make_value(times: np.ndarray, start_value: float = 0.0) -> AxisArray:
    data = (start_value + np.arange(len(times), dtype=float))[:, None]
    return AxisArray(
        data=data,
        dims=["time", "ch"],
        axes={
            "time": AxisArray.CoordinateAxis(data=np.asarray(times, dtype=float), dims=["time"]),
            "ch": AxisArray.CoordinateAxis(data=np.array(["value"]), dims=["ch"]),
        },
        key="value",
    )


def test_seqseq_sampler_aligns_one_sample_mismatch():
    sampler = SeqSeqSampler(max_buffer_dur=1.0)
    signal = _make_signal(n_times=5, fs=10.0)
    value = _make_value(np.array([0.08, 0.18, 0.28, 0.38]), start_value=10.0)
    trigger = SampleTriggerMessage(timestamp=0.0, period=(0.0, 0.25))

    sampler._update_buffer(signal, "signal")
    sampler._update_buffer(value, "value")

    sample, keep_waiting = sampler._process_trigger(trigger)

    assert keep_waiting is False
    assert sample is not None
    assert trigger.value is None
    assert sample.data.shape == (2, 1)
    assert sample.attrs["trigger"].value.data.shape == (2, 1)
    np.testing.assert_allclose(sample.axes["time"].value(np.arange(sample.data.shape[0])), np.array([0.1, 0.2]))
    np.testing.assert_allclose(sample.attrs["trigger"].value.axes["time"].data, np.array([0.08, 0.18]))
    np.testing.assert_allclose(sample.data[:, 0], np.array([1.0, 2.0]))
    np.testing.assert_allclose(sample.attrs["trigger"].value.data[:, 0], np.array([10.0, 11.0]))


def test_seqseq_sampler_requeues_trigger_until_window_is_filled():
    sampler = SeqSeqSampler(max_buffer_dur=1.0)
    trigger = SampleTriggerMessage(timestamp=0.0, period=(0.0, 0.25))

    sampler._update_buffer(_make_signal(n_times=2, fs=10.0), "signal")
    sampler._update_buffer(_make_value(np.array([0.0, 0.1])), "value")

    async def _exercise():
        await sampler.enqueue_trigger(trigger)
        first = await anext(sampler)
        qsize_after_first = sampler._trig_queue.qsize()

        sampler._update_buffer(_make_signal(n_times=2, fs=10.0, offset=0.2, start_value=2.0), "signal")
        sampler._update_buffer(_make_value(np.array([0.2, 0.3]), start_value=2.0), "value")

        second = await anext(sampler)
        return first, qsize_after_first, second, sampler._trig_queue.qsize()

    first, qsize_after_first, second, qsize_after_second = asyncio.run(_exercise())

    assert first is None
    assert qsize_after_first == 1
    assert second is not None
    assert second.data.shape == (3, 1)
    assert second.attrs["trigger"].value.data.shape == (3, 1)
    assert qsize_after_second == 0


def test_seqseq_sampler_drops_gap_trigger_without_requeue():
    sampler = SeqSeqSampler(max_buffer_dur=1.0)
    trigger = SampleTriggerMessage(timestamp=0.05, period=(0.0, 0.04))

    sampler._update_buffer(_make_signal(n_times=3, fs=10.0), "signal")
    sampler._update_buffer(_make_value(np.array([0.0, 0.1, 0.2])), "value")

    async def _exercise():
        await sampler.enqueue_trigger(trigger)
        first = await anext(sampler)
        second = await anext(sampler)
        return first, second, sampler._trig_queue.qsize()

    first, second, qsize = asyncio.run(_exercise())

    assert first is None
    assert second is None
    assert qsize == 0


def test_seqseq_sampler_drops_large_length_mismatch_without_requeue():
    sampler = SeqSeqSampler(max_buffer_dur=1.0)
    trigger = SampleTriggerMessage(timestamp=0.0, period=(0.0, 0.45))

    sampler._update_buffer(_make_signal(n_times=6, fs=10.0), "signal")
    sampler._update_buffer(_make_value(np.array([0.05, 0.25, 0.45])), "value")

    async def _exercise():
        await sampler.enqueue_trigger(trigger)
        first = await anext(sampler)
        second = await anext(sampler)
        return first, second, sampler._trig_queue.qsize()

    first, second, qsize = asyncio.run(_exercise())

    assert first is None
    assert second is None
    assert qsize == 0
