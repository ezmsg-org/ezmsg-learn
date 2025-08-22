import torch
import torch.nn


class MLP(torch.nn.Module):
    """
    A simple Multi-Layer Perceptron (MLP) model. Adapted from Ezmsg MLP.

    Attributes:
        feature_extractor (torch.nn.Sequential): The sequential feature extractor part of the MLP.
        heads (torch.nn.ModuleDict): A dictionary of output linear layers for each output head.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int | list[int],
        num_layers: int | None = None,
        output_heads: int | dict[str, int] = 2,
        norm_layer: str | None = None,
        activation_layer: str | None = "ReLU",
        inplace: bool | None = None,
        bias: bool = True,
        dropout: float = 0.0,
    ):
        """
        Initialize the MLP model.
        Args:
            input_size (int): The size of the input features.
            hidden_size (int | list[int]): The sizes of the hidden layers. If a list, num_layers must be None or the length
                of the list. If a single integer, num_layers must be specified and determines the number of hidden layers.
            num_layers (int, optional): The number of hidden layers. Length of hidden_size if None. Default is None.
            output_heads (int | dict[str, int], optional): Number of output features or classes if single head output or a
                dictionary mapping head names to output sizes if multi-head output. Default is 2 (single head).
            norm_layer (str, optional): A normalization layer to be applied after each linear layer. Default is None.
                Common choices are "BatchNorm1d" or "LayerNorm".
            activation_layer (str, optional): An activation function to be applied after each normalization
                layer. Default is "ReLU".
            inplace (bool, optional): Whether the activation function is performed in-place. Default is None.
            bias (bool, optional): Whether to use bias in the linear layers. Default is True.
            dropout (float, optional): The dropout rate to be applied after each linear layer. Default is 0.0.
        """
        super().__init__()
        if isinstance(hidden_size, int):
            if num_layers is None:
                raise ValueError(
                    "If hidden_size is an integer, num_layers must be specified."
                )
            hidden_size = [hidden_size] * num_layers
        if len(hidden_size) == 0:
            raise ValueError("hidden_size must have at least one element")
        if any(not isinstance(x, int) for x in hidden_size):
            raise ValueError("hidden_size must contain only integers")
        if num_layers is not None and len(hidden_size) != num_layers:
            raise ValueError(
                "Length of hidden_size must match num_layers if num_layers is specified."
            )

        params = {} if inplace is None else {"inplace": inplace}

        layers = []
        in_dim = input_size

        def _get_layer_class(layer_name: str):
            if layer_name is not None and "torch.nn" in layer_name:
                return getattr(torch.nn, layer_name.rsplit(".", 1)[1])
            return None

        norm_layer_class = _get_layer_class(norm_layer)
        activation_layer_class = _get_layer_class(activation_layer)
        for hidden_dim in hidden_size[:-1]:
            layers.append(torch.nn.Linear(in_dim, hidden_dim, bias=bias))
            if norm_layer_class is not None:
                layers.append(norm_layer_class(hidden_dim))
            if activation_layer_class is not None:
                layers.append(activation_layer_class(**params))
            layers.append(torch.nn.Dropout(dropout, **params))
            in_dim = hidden_dim

        layers.append(torch.nn.Linear(in_dim, hidden_size[-1], bias=bias))

        self.feature_extractor = torch.nn.Sequential(*layers)

        if isinstance(output_heads, int):
            output_heads = {"output": output_heads}
        self.heads = torch.nn.ModuleDict(
            {
                name: torch.nn.Linear(hidden_size[-1], output_size)
                for name, output_size in output_heads.items()
            }
        )

    @classmethod
    def infer_config_from_state_dict(cls, state_dict: dict) -> dict[str, int | float]:
        """
        Infer the configuration from the state dict.

        Args:
            state_dict: The state dict of the model.

        Returns:
            dict[str, int | float]: A dictionary containing the inferred configuration.
        """
        input_size = state_dict["feature_extractor.0.weight"].shape[1]
        hidden_size = [
            param.shape[0]
            for key, param in state_dict.items()
            if key.startswith("feature_extractor.") and key.endswith(".weight")
        ]
        output_heads = {
            key.split(".")[1]: param.shape[0]
            for key, param in state_dict.items()
            if key.startswith("heads.") and key.endswith(".bias")
        }

        return {
            "input_size": input_size,
            "hidden_size": hidden_size,
            "output_heads": output_heads,
        }

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Forward pass through the MLP.

        Args:
            x (torch.Tensor): Input tensor of shape (batch, seq_len, input_size).

        Returns:
            dict[str, torch.Tensor]: A dictionary mapping head names to output tensors.
        """
        x = self.feature_extractor(x)
        return {name: head(x) for name, head in self.heads.items()}
