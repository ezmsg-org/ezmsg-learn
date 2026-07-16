import numpy as np
import torch
from ezmsg.util.messages.axisarray import AxisArray
from torchaudio.functional import edit_distance

from .base import BaseAssessSettings, BaseAssessUnit


class ErrorRateSettings(BaseAssessSettings):
    seq_axis: str = "time"
    batch_axis: str = "batch"


class ErrorRate(BaseAssessUnit):
    """Compute per-sequence error rate using torchaudio edit distance.

    Subscribes to ground-truth and prediction ``AxisArray`` messages, pairs them
    in FIFO order, computes error rate (edit_distance / gt_length) for each
    sequence, and publishes results on ``OUTPUT_METRIC``.

    When the data carries a ``batch_axis`` dimension the unit uses torchaudio's
    batched ``edit_distance``; otherwise a single 1-D sequence is assumed.
    """

    SETTINGS = ErrorRateSettings

    async def _assess(self, gt: AxisArray, pred: AxisArray) -> AxisArray | None:
        if self.SETTINGS.seq_axis not in gt.dims:
            return None

        batch_given = self.SETTINGS.batch_axis in gt.dims

        if batch_given:
            ers = self._assess_batched(gt, pred, self.SETTINGS.batch_axis)
        else:
            ers = self._assess_single(gt, pred)

        return AxisArray(data=ers, dims=[self.SETTINGS.batch_axis])

    # ------------------------------------------------------------------
    @staticmethod
    def _assess_batched(gt: AxisArray, pred: AxisArray, batch_axis: str) -> AxisArray:

        gt_data = gt.data
        gt_lens = gt.attrs["trigger_len"]
        pred_data = pred.data
        pred_lens = pred.attrs["trigger_len"]

        b_idx = gt.dims.index(batch_axis)

        batch_size = gt_data.shape[b_idx]

        device = torch.device("cpu")
        ers = torch.zeros_like(batch_size, dtype=torch.float32, device=device)

        for b in range(batch_size):
            gt_seq = torch.from_numpy(gt_data.take(b, axis=b_idx)).to(torch.int64).flatten()
            pred_seq = torch.from_numpy(pred_data.take(b, axis=b_idx)).to(torch.int64).flatten()

            ed = edit_distance(gt_seq[: gt_lens[b]], pred_seq[: pred_lens[b]])
            ers[b] = float(ed) / len(gt_seq)

        return ers.cpu().numpy().astype(float)

    # ------------------------------------------------------------------
    @staticmethod
    def _assess_single(gt: AxisArray, pred: AxisArray) -> AxisArray:
        gt_seq = torch.from_numpy(gt.data).to(torch.int64)
        pred_seq = torch.from_numpy(pred.data).to(torch.int64)

        ed = edit_distance(gt_seq, pred_seq)
        er = float(ed) / len(gt_seq)

        return np.array([er], dtype=float)
