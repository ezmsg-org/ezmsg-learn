import torch
from ezmsg.util.messages.axisarray import AxisArray
from torchaudio.functional import edit_distance

from .base import BaseAssessSettings, BaseAssessUnit


class ErrorRateSettings(BaseAssessSettings):
    seq_axis: str = "time"


class ErrorRate(BaseAssessUnit):
    """Compute per-sequence error rate using torchaudio edit distance.

    Subscribes to ground-truth and prediction ``AxisArray`` messages, pairs them
    in FIFO order. ``BaseAssessUnit`` handles looping over any batch axis, so
    ``_assess`` here always receives a single, unbatched ``gt``/``pred``
    sequence pair and returns their error rate (edit_distance / gt_length).
    """

    SETTINGS = ErrorRateSettings

    async def _assess(self, gt: AxisArray, pred: AxisArray) -> float | None:
        if self.SETTINGS.seq_axis not in gt.dims:
            return None

        gt_seq = torch.from_numpy(gt.data).to(torch.int64).flatten()
        pred_seq = torch.from_numpy(pred.data).to(torch.int64).flatten()

        gt_len = gt.attrs.get("trigger_len")
        pred_len = pred.attrs.get("output_len")
        if gt_len is not None:
            gt_seq = gt_seq[:gt_len]
        if pred_len is not None:
            pred_seq = pred_seq[:pred_len]

        ed = edit_distance(gt_seq, pred_seq)
        return float(ed) / len(gt_seq)
