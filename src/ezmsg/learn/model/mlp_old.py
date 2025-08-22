import torch
import torch.nn


class MLP(torch.nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: list[int],
        norm_layer: torch.nn.Module | None = None,
        activation_layer: torch.nn.Module | None = torch.nn.ReLU,
        inplace: bool | None = None,
        bias: bool = True,
        dropout: float = 0.0,
    ):
        """
        Copy-pasted from torchvision MLP

        :param in_channels: Number of input channels
        :param hidden_channels: List of the hidden channel dimensions
        :param norm_layer: Norm layer that will be stacked on top of the linear layer. If None this layer won’t be used.
        :param activation_layer: Activation function which will be stacked on top of the normalization layer
          (if not None), otherwise on top of the linear layer. If None this layer won’t be used.
        :param inplace: Parameter for the activation layer, which can optionally do the operation in-place.
          Default is None, which uses the respective default values of the activation_layer and Dropout layer.
        :param bias: Whether to use bias in the linear layer.
        :param dropout: The probability for the dropout layer.
        """
        if len(hidden_channels) == 0:
            raise ValueError("hidden_channels must have at least one element")
        if any(not isinstance(x, int) for x in hidden_channels):
            raise ValueError("hidden_channels must contain only integers")

        params = {} if inplace is None else {"inplace": inplace}

        layers = []
        in_dim = in_channels
        for hidden_dim in hidden_channels[:-1]:
            layers.append(torch.nn.Linear(in_dim, hidden_dim, bias=bias))
            if norm_layer is not None:
                layers.append(norm_layer(hidden_dim))
            if activation_layer is not None:
                layers.append(activation_layer(**params))
            layers.append(torch.nn.Dropout(dropout, **params))
            in_dim = hidden_dim

        layers.append(torch.nn.Linear(in_dim, hidden_channels[-1], bias=bias))

        super().__init__(*layers)
