from typing import Optional

import torch


class RNNModel(torch.nn.Module):
    """
    Recurrent neural network supporting GRU, LSTM, and vanilla RNN (tanh/relu).

    Attributes:
        input_size (int): Number of input features per time step.
        hidden_size (int): Number of hidden units in the RNN cell.
        num_layers (int, optional): Number of RNN layers. Default is 1.
        output_size (int | dict[str, int], optional): Number of output features or classes if single head output or a
            dictionary mapping head names to output sizes if multi-head output. Default is 2 (single head).
        dropout (float, optional): Dropout rate applied after input and RNN output. Default is 0.3.
        rnn_type (str, optional): Type of RNN cell to use: 'GRU', 'LSTM', 'RNN-Tanh', 'RNN-ReLU'. Default is 'GRU'.

    Returns:
        dict[str, torch.Tensor]: Dictionary of decoded predictions mapping head names to tensors of shape
            (batch, seq_len, output_size). If single head output, the key is "output".
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        output_size: int | dict[str, int] = 2,
        dropout: float = 0.3,
        rnn_type: str = "GRU",
    ):
        super().__init__()
        self.linear_embeddings = torch.nn.Linear(input_size, input_size)
        self.dropout_input = torch.nn.Dropout(dropout)

        rnn_klass_str = rnn_type.upper().split("-")[0]
        if rnn_klass_str not in ["GRU", "LSTM", "RNN"]:
            raise ValueError(f"Unrecognized rnn_type: {rnn_type}")
        rnn_klass = {"GRU": torch.nn.GRU, "LSTM": torch.nn.LSTM, "RNN": torch.nn.RNN}[
            rnn_klass_str
        ]
        rnn_kwargs = {}
        if rnn_klass_str == "RNN":
            rnn_kwargs["nonlinearity"] = rnn_type.lower().split("-")[-1]
        self.rnn = rnn_klass(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            **rnn_kwargs,
        )
        self.rnn_type = rnn_klass_str

        self.output_dropout = torch.nn.Dropout(dropout)
        if isinstance(output_size, int):
            output_size = {"output": output_size}
        self.heads = torch.nn.ModuleDict(
            {
                name: torch.nn.Linear(hidden_size, size)
                for name, size in output_size.items()
            }
        )

    @classmethod
    def infer_config_from_state_dict(
        cls, state_dict: dict, rnn_type: str = "GRU"
    ) -> dict[str, int | float]:
        """
        This method is specific to each processor.

        Args:
            state_dict: The state dict of the model.
            rnn_type: The type of RNN used in the model (e.g., 'GRU', 'LSTM', 'RNN-Tanh', 'RNN-ReLU').

        Returns:
            A dictionary of model parameters obtained from the state dict.

        """
        output_size = {
            key.split(".")[1]: param.shape[0]
            for key, param in state_dict.items()
            if key.startswith("heads.") and key.endswith(".bias")
        }

        return {
            # Infer input_size from linear_embeddings.weight (shape: [input_size, input_size])
            "input_size": state_dict["linear_embeddings.weight"].shape[1],
            # Infer hidden_size from rnn.weight_ih_l0 (shape: [hidden_size * 3, input_size])
            "hidden_size": state_dict["rnn.weight_ih_l0"].shape[0]
            // cls._get_gate_count(rnn_type),
            # Infer num_layers by counting rnn layers in state_dict (e.g., weight_ih_l<k>)
            "num_layers": sum(1 for key in state_dict if "rnn.weight_ih_l" in key),
            "output_size": output_size,
        }

    @staticmethod
    def _get_gate_count(rnn_type: str) -> int:
        if rnn_type.upper() == "GRU":
            return 3
        elif rnn_type.upper() == "LSTM":
            return 4
        elif rnn_type.upper().startswith("RNN"):
            return 1
        else:
            raise ValueError(f"Unsupported rnn_type: {rnn_type}")

    def init_hidden(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """
        Initialize the hidden state for the RNN.
        Args:
            batch_size (int): Size of the batch.
            device (torch.device): Device to place the hidden state on (e.g., 'cpu' or 'cuda').
        Returns:
            torch.Tensor | tuple[torch.Tensor, torch.Tensor]: Initial hidden state for the RNN.
                For LSTM, returns a tuple of (h_n, c_n) where h_n is the hidden state and c_n is the cell state.
                For GRU or vanilla RNN, returns just h_n.
        """
        shape = (self.rnn.num_layers, batch_size, self.rnn.hidden_size)
        if self.rnn_type == "LSTM":
            return (
                torch.zeros(shape, device=device),
                torch.zeros(shape, device=device),
            )
        else:
            return torch.zeros(shape, device=device)

    def forward(
        self,
        x: torch.Tensor,
        input_lens: Optional[torch.Tensor] = None,
        hx: Optional[torch.Tensor | tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor | tuple]:
        """
        Forward pass through the RNN model.
        Args:
            x (torch.Tensor): Input tensor of shape (batch, seq_len, input_size).
            input_lens (Optional[torch.Tensor]): Optional tensor of lengths for each sequence in the batch.
                If provided, sequences will be packed before passing through the RNN.
            hx (Optional[torch.Tensor | tuple[torch.Tensor, torch.Tensor]]): Optional initial hidden state for the RNN.
        Returns:
            tuple[dict[str, torch.Tensor], torch.Tensor | tuple]:
                A dictionary mapping head names to output tensors of shape (batch, seq_len, output_size).
                If the RNN is LSTM, the second element is the hidden state (h_n, c_n) or just h_n if GRU.
        """
        x = self.linear_embeddings(x)
        x = self.dropout_input(x)
        total_length = x.shape[1]
        if input_lens is not None:
            x = torch.nn.utils.rnn.pack_padded_sequence(
                x, input_lens, batch_first=True, enforce_sorted=False
            )
        x_out, hx_out = self.rnn(x, hx)
        if input_lens is not None:
            x_out, _ = torch.nn.utils.rnn.pad_packed_sequence(
                x_out, batch_first=True, total_length=total_length
            )
        x_out = self.output_dropout(x_out)
        return {name: head(x_out) for name, head in self.heads.items()}, hx_out
