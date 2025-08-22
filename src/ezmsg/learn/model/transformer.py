from typing import Optional

import torch


class TransformerModel(torch.nn.Module):
    """
    Transformer-based encoder (optional decoder) neural network.

    If `decoder_layers > 0`, the model includes a Transformer decoder. In this case, the `tgt` argument must be
    provided: during training, it is typically the ground-truth target sequence (i.e. teacher forcing); during
    inference, it can be constructed autoregressively from previous predictions.

    Attributes:
        input_size (int): Number of input features per time step.
        hidden_size (int): Dimensionality of the transformer model.
        encoder_layers (int, optional): Number of transformer encoder layers. Default is 1.
        decoder_layers (int, optional): Number of transformer decoder layers. Default is 0.
        output_size (int | dict[str, int], optional): Number of output features or classes if single head output, or a
            dictionary mapping head names to output sizes if multi-head output. Default is 2 (single head).
        dropout (float, optional): Dropout rate applied after input and transformer output. Default is 0.3.
        attention_heads (int, optional): Number of attention heads in the transformer. Default is 4.
        max_seq_len (int, optional): Maximum sequence length for positional embeddings. Default is 512.

    Returns:
        dict[str, torch.Tensor]: Dictionary of decoded predictions mapping head names to tensors of shape
            (batch, seq_len, output_size). If single head output, the key is "output".
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        encoder_layers: int = 1,
        decoder_layers: int = 0,
        output_size: int | dict[str, int] = 2,
        dropout: float = 0.3,
        attention_heads: int = 4,
        max_seq_len: int = 512,
        autoregressive_head: str | None = None,
    ):
        super().__init__()

        self.decoder_layers = decoder_layers
        self.hidden_size = hidden_size

        if isinstance(output_size, int):
            autoregressive_size = output_size
        else:
            autoregressive_size = list(output_size.values())[0]
        if isinstance(output_size, dict):
            autoregressive_size = output_size.get(
                autoregressive_head, autoregressive_size
            )
        self.start_token = torch.nn.Parameter(torch.zeros(1, 1, autoregressive_size))
        self.output_to_hidden = torch.nn.Linear(autoregressive_size, hidden_size)

        self.input_proj = torch.nn.Linear(input_size, hidden_size)
        self.pos_embedding = torch.nn.Embedding(max_seq_len, hidden_size)
        self.dropout = torch.nn.Dropout(dropout)

        self.encoder = torch.nn.TransformerEncoder(
            torch.nn.TransformerEncoderLayer(
                d_model=hidden_size,
                nhead=attention_heads,
                dim_feedforward=hidden_size * 4,
                dropout=dropout,
                batch_first=True,
            ),
            num_layers=encoder_layers,
        )

        self.decoder = None
        if decoder_layers > 0:
            self.decoder = torch.nn.TransformerDecoder(
                torch.nn.TransformerDecoderLayer(
                    d_model=hidden_size,
                    nhead=attention_heads,
                    dim_feedforward=hidden_size * 4,
                    dropout=dropout,
                    batch_first=True,
                ),
                num_layers=decoder_layers,
            )

        if isinstance(output_size, int):
            output_size = {"output": output_size}
        self.heads = torch.nn.ModuleDict(
            {
                name: torch.nn.Linear(hidden_size, out_dim)
                for name, out_dim in output_size.items()
            }
        )

    @classmethod
    def infer_config_from_state_dict(cls, state_dict: dict) -> dict[str, int | float]:
        # Infer output size from heads.<name>.bias (shape: [output_size])
        output_size = {
            key.split(".")[1]: param.shape[0]
            for key, param in state_dict.items()
            if key.startswith("heads.") and key.endswith(".bias")
        }

        return {
            # Infer input_size from input_proj.weight (shape: [hidden_size, input_size])
            "input_size": state_dict["input_proj.weight"].shape[1],
            # Infer hidden_size from input_proj.weight (shape: [hidden_size, input_size])
            "hidden_size": state_dict["input_proj.weight"].shape[0],
            "output_size": output_size,
            # Infer encoder_layers from transformer layers in state_dict
            "encoder_layers": len(
                [k for k in state_dict if k.startswith("encoder.layers")]
            ),
            # Infer decoder_layers from transformer decoder layers in state_dict
            "decoder_layers": len(
                {k.split(".")[2] for k in state_dict if k.startswith("decoder.layers")}
            )
            if any(k.startswith("decoder.layers") for k in state_dict)
            else 0,
        }

    def forward(
        self,
        src: torch.Tensor,
        tgt: Optional[torch.Tensor] = None,
        src_mask: Optional[torch.Tensor] = None,
        tgt_mask: Optional[torch.Tensor] = None,
        start_pos: int = 0,
    ) -> dict[str, torch.Tensor]:
        """
        Forward pass through the transformer model.
        Args:
            src (torch.Tensor): Input tensor of shape (batch, seq_len, input_size).
            tgt (Optional[torch.Tensor]): Target tensor for decoder, shape (batch, seq_len, input_size).
                Required if `decoder_layers > 0`. In training, this can be the ground-truth target sequence
                (i.e. teacher forcing). During inference, this is constructed autoregressively.
            src_mask (Optional[torch.Tensor]): Optional attention mask for the encoder input. Should be broadcastable
                to shape (batch, seq_len, seq_len) or (seq_len, seq_len).
            tgt_mask (Optional[torch.Tensor]): Optional attention mask for the decoder input. Used to enforce causal
                decoding (i.e. autoregressive generation) during training or inference.
            start_pos (int): Starting offset for positional embeddings. Used for streaming inference to maintain
                correct positional indices. Default is 0.
        Returns:
            dict[str, torch.Tensor]: Dictionary of output tensors each output head, each with shape (batch, seq_len,
                output_size).
        """
        B, T, _ = src.shape
        device = src.device

        x = self.input_proj(src)
        pos_ids = torch.arange(start_pos, start_pos + T, device=device).expand(B, T)
        x = x + self.pos_embedding(pos_ids)
        x = self.dropout(x)

        memory = self.encoder(x, mask=src_mask)

        if self.decoder is not None:
            if tgt is None:
                tgt = self.start_token.expand(B, -1, -1).to(device)
            tgt_proj = self.output_to_hidden(tgt)
            tgt_pos_ids = torch.arange(tgt.shape[1], device=device).expand(
                B, tgt.shape[1]
            )
            tgt_proj = tgt_proj + self.pos_embedding(tgt_pos_ids)
            tgt_proj = self.dropout(tgt_proj)
            out = self.decoder(
                tgt_proj,
                memory,
                tgt_mask=tgt_mask,
                memory_mask=src_mask,
            )
        else:
            out = memory

        return {name: head(out) for name, head in self.heads.items()}
