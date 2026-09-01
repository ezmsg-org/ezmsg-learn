import asyncio
import itertools
import typing
from abc import ABC, abstractmethod
from collections.abc import Generator

import ezmsg.core as ez
import numpy as np
from ezmsg.util.messages.axisarray import AxisArray
from ezmsg.util.messages.util import replace


class BaseAssessSettings(ez.Settings):
    log_level: str = "INFO"
    batch_axis: str = "batch"


GTType = typing.TypeVar("GTType", default=AxisArray)
PredType = typing.TypeVar("PredType", default=AxisArray)


class BaseAssessUnit(ez.Unit, ABC, typing.Generic[GTType, PredType]):
    """Abstract ezmsg Unit for performance assessment.

    Subscribes to two streams (ground truth + predictions), pairs them in
    FIFO order, loops over the ``batch_axis`` (if present on either side),
    and calls the abstract ``_assess`` once per single, unbatched sequence
    pair. The per-item results are collected into one ``AxisArray`` (dims
    ``[batch_axis]``) published on ``OUTPUT_METRIC``.
    """

    SETTINGS = BaseAssessSettings

    INPUT_GT = ez.InputStream(AxisArray, leaky=True)
    INPUT_PRED = ez.InputStream(AxisArray, leaky=True)
    OUTPUT_METRIC = ez.OutputStream(AxisArray)

    _gt_queue: asyncio.Queue[GTType]
    _pred_queue: asyncio.Queue[PredType]
    _result_queue: asyncio.Queue[AxisArray]
    _assessing: asyncio.Lock

    async def initialize(self) -> None:
        self._gt_queue = asyncio.Queue()
        self._pred_queue = asyncio.Queue()
        self._result_queue = asyncio.Queue()
        self._assessing = asyncio.Lock()

    @ez.subscriber(INPUT_GT)
    async def on_gt(self, msg: GTType) -> None:
        await self._gt_queue.put(msg)
        await self._try_assess()

    @ez.subscriber(INPUT_PRED)
    async def on_pred(self, msg: PredType) -> None:
        await self._pred_queue.put(msg)
        await self._try_assess()

    async def _try_assess(self) -> None:
        async with self._assessing:
            while True:
                try:
                    gt = self._gt_queue.get_nowait()
                    pred = self._pred_queue.get_nowait()
                except asyncio.QueueEmpty:
                    break

                metrics = []
                for gt_item, pred_item in self._iter_paired(gt, pred):
                    metric = await self._assess(gt_item, pred_item)
                    if metric is not None:
                        metrics.append(metric)

                if metrics:
                    result = AxisArray(
                        data=np.asarray(metrics, dtype=float),
                        dims=[self.SETTINGS.batch_axis],
                    )
                    await self._result_queue.put(result)

    def _iter_paired(self, gt: GTType, pred: PredType) -> Generator[tuple[GTType, PredType], None, None]:
        """Yield one (gt_item, pred_item) pair per batch item.

        If neither operand carries ``batch_axis``, yields the pair once,
        unchanged. Otherwise, the batch sizes (1 for a side that isn't
        batched) must agree; a mismatch raises ``ValueError``. Per-item
        length-like attrs (e.g. arrays of shape ``(batch_size,)``) are
        indexed down to a scalar for whichever side was actually batched.
        """
        axis = self.SETTINGS.batch_axis
        gt_batched = axis in gt.dims
        pred_batched = axis in pred.dims

        if not gt_batched and not pred_batched:
            yield gt, pred
            return

        gt_size = gt.data.shape[gt.get_axis_idx(axis)] if gt_batched else 1
        pred_size = pred.data.shape[pred.get_axis_idx(axis)] if pred_batched else 1
        if gt_size != pred_size:
            raise ValueError(f"Incompatible batch sizes: gt has {gt_size}, pred has {pred_size}")

        gt_items = gt.iter_over_axis(axis) if gt_batched else itertools.repeat(gt)
        pred_items = pred.iter_over_axis(axis) if pred_batched else itertools.repeat(pred)

        for idx, (gt_item, pred_item) in enumerate(zip(gt_items, pred_items)):
            if gt_batched:
                gt_item = self._slice_attrs(gt_item, idx, gt_size)
            if pred_batched:
                pred_item = self._slice_attrs(pred_item, idx, pred_size)
            yield gt_item, pred_item

    @staticmethod
    def _slice_attrs(item: GTType, idx: int, batch_size: int) -> GTType:
        """Index any per-batch-item attrs down to a scalar for this item.

        An attr value is treated as per-batch-item metadata (and indexed with
        ``[idx]``) only if it's a list/tuple/ndarray whose length equals
        ``batch_size``; everything else passes through unchanged.
        """
        attrs = {
            key: (value[idx] if isinstance(value, (list, tuple, np.ndarray)) and len(value) == batch_size else value)
            for key, value in item.attrs.items()
        }
        return replace(item, attrs=attrs)

    @ez.publisher(OUTPUT_METRIC)
    async def emit(self):
        while True:
            metric = await self._result_queue.get()
            log_fn = getattr(ez.logger, self.SETTINGS.log_level.lower(), ez.logger.info)
            log_fn(f"AssessUnit: {metric.data}")
            yield self.OUTPUT_METRIC, metric

    @abstractmethod
    async def _assess(self, gt: GTType, pred: PredType) -> float | None: ...
