import asyncio
import typing
from abc import ABC
from abc import abstractmethod

import ezmsg.core as ez
from ezmsg.util.messages.axisarray import AxisArray


class BaseAssessSettings(ez.Settings):
    log_level: str = "INFO"


GTType = typing.TypeVar("GTType", default=AxisArray)
PredType = typing.TypeVar("PredType", default=AxisArray)


class BaseAssessUnit(ez.Unit, ABC, typing.Generic[GTType, PredType]):
    """Abstract ezmsg Unit for performance assessment.

    Subscribes to two streams (ground truth + predictions), pairs them in
    FIFO order, runs a performance assessment function, and publishes the
    result as an ``AxisArray`` on ``OUTPUT_METRIC``.
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
                result = await self._assess(gt, pred)
                if result is not None:
                    await self._result_queue.put(result)

    @ez.publisher(OUTPUT_METRIC)
    async def emit(self):
        while True:
            metric = await self._result_queue.get()
            log_fn = getattr(ez.logger, self.SETTINGS.log_level.lower(), ez.logger.info)
            log_fn(f"AssessUnit: {metric.data}")
            yield self.OUTPUT_METRIC, metric

    @abstractmethod
    async def _assess(self, gt: GTType, pred: PredType) -> AxisArray | None: ...
