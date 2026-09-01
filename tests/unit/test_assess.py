import numpy as np
import pytest
from ezmsg.util.messages.axisarray import AxisArray

from ezmsg.learn.assess.base import BaseAssessSettings, BaseAssessUnit
from ezmsg.learn.assess.error_rate import ErrorRate, ErrorRateSettings


class _RecordingAssess(BaseAssessUnit):
    """Minimal concrete BaseAssessUnit used to exercise batch-pairing logic in isolation."""

    async def _assess(self, gt: AxisArray, pred: AxisArray) -> float | None:
        return 1.0


def _batched(data: np.ndarray, batch_size: int, *, dims=("time",), attrs=None) -> AxisArray:
    return AxisArray(
        data=data,
        dims=["batch", *dims],
        axes={"batch": AxisArray.CoordinateAxis(data=np.arange(batch_size), dims=["batch"])},
        attrs=attrs or {},
    )


def _unbatched(data: np.ndarray, *, dims=("time",), attrs=None) -> AxisArray:
    return AxisArray(data=data, dims=list(dims), attrs=attrs or {})


def test_iter_paired_promotes_unbatched_gt_against_batch_of_one_pred():
    # Reproduces the shapes from the original crash: gt has no batch axis at
    # all, pred is batched with size 1 and carries a padded-length attr.
    unit = _RecordingAssess(settings=BaseAssessSettings())
    gt = _unbatched(np.array([1, 2, 3, 4, 5], dtype=np.int64))
    pred = _batched(
        np.array([[1, 2, 9, 4, 5, 0, 0]], dtype=np.int64),
        batch_size=1,
        attrs={"output_len": np.array([5])},
    )

    pairs = list(unit._iter_paired(gt, pred))

    assert len(pairs) == 1
    gt_item, pred_item = pairs[0]
    assert gt_item.dims == ["time"]
    assert pred_item.dims == ["time"]
    assert pred_item.data.shape == (7,)
    assert pred_item.attrs["output_len"] == 5


def test_iter_paired_raises_on_batch_size_mismatch_with_unbatched():
    unit = _RecordingAssess(settings=BaseAssessSettings())
    gt = _unbatched(np.array([1, 2, 3], dtype=np.int64))
    pred = _batched(np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int64), batch_size=2)

    with pytest.raises(ValueError):
        list(unit._iter_paired(gt, pred))


def test_iter_paired_raises_on_batch_size_mismatch_both_batched():
    unit = _RecordingAssess(settings=BaseAssessSettings())
    gt = _batched(np.array([[1, 2, 3]], dtype=np.int64), batch_size=1)
    pred = _batched(np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int64), batch_size=2)

    with pytest.raises(ValueError):
        list(unit._iter_paired(gt, pred))


def test_iter_paired_passes_through_when_both_unbatched():
    unit = _RecordingAssess(settings=BaseAssessSettings())
    gt = _unbatched(np.array([1, 2, 3], dtype=np.int64))
    pred = _unbatched(np.array([1, 2, 3], dtype=np.int64))

    assert list(unit._iter_paired(gt, pred)) == [(gt, pred)]


def test_iter_paired_indexes_attrs_per_item_when_both_batched():
    unit = _RecordingAssess(settings=BaseAssessSettings())
    gt = _batched(
        np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.int64),
        batch_size=3,
        attrs={"trigger_len": np.array([1, 2, 3])},
    )
    pred = _batched(
        np.array([[9, 9, 9], [8, 8, 8], [7, 7, 7]], dtype=np.int64),
        batch_size=3,
        attrs={"output_len": np.array([3, 2, 1])},
    )

    pairs = list(unit._iter_paired(gt, pred))

    assert len(pairs) == 3
    for i, (gt_item, pred_item) in enumerate(pairs):
        assert np.array_equal(gt_item.data, gt.data[i])
        assert np.array_equal(pred_item.data, pred.data[i])
        assert gt_item.attrs["trigger_len"] == gt.attrs["trigger_len"][i]
        assert pred_item.attrs["output_len"] == pred.attrs["output_len"][i]


async def test_error_rate_handles_unbatched_gt_with_batch_of_one_pred():
    # End-to-end: the exact gt/pred shapes from the original crash traceback.
    unit = ErrorRate(settings=ErrorRateSettings())
    gt = _unbatched(np.array([1, 2, 3, 4, 5], dtype=np.int64))
    pred = _batched(
        np.array([[1, 2, 9, 4, 5, 0, 0]], dtype=np.int64),
        batch_size=1,
        attrs={"output_len": np.array([5])},
    )

    metrics = [
        await unit._assess(gt_item, pred_item)
        for gt_item, pred_item in unit._iter_paired(gt, pred)
    ]

    assert metrics == pytest.approx([0.2])
