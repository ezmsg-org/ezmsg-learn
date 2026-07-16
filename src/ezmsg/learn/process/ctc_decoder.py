import ezmsg.core as ez
import numpy as np
import torch
from ezmsg.baseproc import (
    BaseStatefulTransformer,
    BaseTransformerUnit,
    processor_state,
)
from ezmsg.util.messages.axisarray import AxisArray
from ezmsg.util.messages.util import replace


class CTCGreedyDecoderSettings(ez.Settings):
    vocabulary: list[str] | None = None
    """Logit index → label mapping. None = auto-generate."""

    blank_idx: int = 0
    """Index of blank token."""

    single_precision: bool = True
    """Use float32."""


@processor_state
class CTCGreedyDecoderState:
    vocabulary: list[str] | None = None


class CTCGreedyDecoderProcessor(
    BaseStatefulTransformer[CTCGreedyDecoderSettings, AxisArray, AxisArray, CTCGreedyDecoderState]
):
    def _reset_state(self, message: AxisArray) -> None:
        if "batch" not in message.axes:
            raise AssertionError("CTCDecoder requires a 'batch' axis on input messages.")

        num_classes = message.data.shape[-1]

        if self.settings.vocabulary is not None:
            self._state.vocabulary = list(self.settings.vocabulary)
        else:
            self._state.vocabulary = ["<blank>"] + [f"tok_{i}" for i in range(1, num_classes)]

    def _process(self, message: AxisArray) -> list[AxisArray]:
        logits = message.data
        if not isinstance(logits, torch.Tensor):
            dtype = torch.float32 if self.settings.single_precision else torch.float64
            logits = torch.tensor(logits, dtype=dtype)

        data_len = message.attrs["data_len"]
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

        B = message.data.shape[message.get_axis_idx("batch")]
        blank_idx = self.settings.blank_idx
        decoded_sequences = []

        for i in range(B):
            seq_len = int(data_len[i])
            lp = log_probs[i, :seq_len].cpu().numpy()
            preds = np.argmax(lp, axis=-1)
            result = []
            last = -1
            for t in range(seq_len):
                if preds[t] != blank_idx and preds[t] != last:
                    result.append(int(preds[t]))
                last = preds[t]
            decoded_sequences.append(result)

        max_decoded_len = max((len(seq) for seq in decoded_sequences), default=0)
        padded = np.zeros((B, max_decoded_len), dtype=np.int64)
        output_len = np.empty(B, dtype=np.int64)

        for i, seq in enumerate(decoded_sequences):
            seq_len = len(seq)
            output_len[i] = seq_len
            if seq_len > 0:
                padded[i, :seq_len] = seq

        time_axis = AxisArray.CoordinateAxis(
            data=np.arange(max_decoded_len, dtype=np.int64),
            dims=["time"],
        )

        return replace(
            message,
            data=padded,
            dims=["batch", "time"],
            axes={
                "batch": message.axes["batch"],
                "time": time_axis,
            },
            attrs={
                "output_len": output_len,
            },
        )

    def _hash_message(self, message: AxisArray) -> int:
        return hash(tuple(message.dims) + message.data.shape)


class CTCGreedyDecoderUnit(
    BaseTransformerUnit[
        CTCGreedyDecoderSettings,
        AxisArray,
        AxisArray,
        CTCGreedyDecoderProcessor,
    ]
):
    SETTINGS = CTCGreedyDecoderSettings
