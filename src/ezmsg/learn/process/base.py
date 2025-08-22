import inspect
import json
from pathlib import Path
import typing

import ezmsg.core as ez
import torch


class ModelInitMixin:
    """
    Mixin class to support model initialization from:
        1. Setting parameters
        2. Config file
        3. Checkpoint file
    """

    @staticmethod
    def _merge_config(model_kwargs: dict, config) -> None:
        """
        Mutate the model_kwargs dictionary with the config parameters.
        Args:
            model_kwargs: Original to-be-mutated model kwargs.
            config: Update config parameters.

        Returns:
            None because model_kwargs is mutated in place.
        """
        if "model_params" in config:
            config = config["model_params"]
        # Update model_kwargs with config parameters
        for key, value in config.items():
            if key in model_kwargs:
                if model_kwargs[key] != value:
                    ez.logger.warning(
                        f"Config parameter {key} ({value}) differs from settings ({model_kwargs[key]})."
                    )
            else:
                ez.logger.warning(f"Config parameter {key} is not in model_kwargs.")
            model_kwargs[key] = value

    def _filter_model_kwargs(self, model_class, kwargs: dict) -> dict:
        valid_params = inspect.signature(model_class.__init__).parameters
        filtered_out = set(kwargs.keys()) - {k for k in valid_params if k != "self"}
        if filtered_out:
            ez.logger.warning(
                f"Ignoring unexpected model parameters not accepted by {model_class.__name__} constructor: {sorted(filtered_out)}"
            )
        # Keep all valid parameters, including None values, so checkpoint-inferred values can overwrite them
        return {k: v for k, v in kwargs.items() if k in valid_params and k != "self"}

    def _init_model(
        self,
        model_class,
        params: dict[str, typing.Any] | None = None,
        config_path: str | None = None,
        checkpoint_path: str | None = None,
        device: str = "cpu",
        state_dict_prefix: str | None = None,
        weights_only: bool | None = None,
    ) -> torch.nn.Module:
        """
        Args:
            model_class: The class of the model to be initialized.
            params: A dictionary of setting parameters to be used for model initialization.
            config_path: Path to a JSON config file to update model parameters.
            checkpoint_path: Path to a checkpoint file to load model weights and possibly config.

        Returns:
            The initialized model.
        The model will be initialized with the correct config and weights.

        """
        # Model parameters are taken from multiple sources, in ascending priority:
        # 1. Setting parameters
        # 2. Config file if provided
        # 3. "config" entry in checkpoint file if checkpoint file provided and config present
        # 4. Sizes of weights in checkpoint file if provided

        # Get configs from setting params.
        model_kwargs = params or {}
        state_dict = None

        # Check if a config file is provided and if so use that to update kwargs (with warnings).
        if config_path:
            config_path = Path(config_path)
            if not config_path.exists():
                ez.logger.error(f"Config path {config_path} does not exist.")
                raise FileNotFoundError(f"Config path {config_path} does not exist.")
            try:
                with open(config_path, "r") as f:
                    config = json.load(f)
                self._merge_config(model_kwargs, config)
            except Exception as e:
                raise RuntimeError(
                    f"Failed to load config from {config_path}: {str(e)}"
                )

        # If a checkpoint file is provided, load it.
        if checkpoint_path:
            checkpoint_path = Path(checkpoint_path)
            if not checkpoint_path.exists():
                ez.logger.error(f"Checkpoint path {checkpoint_path} does not exist.")
                raise FileNotFoundError(
                    f"Checkpoint path {checkpoint_path} does not exist."
                )
            try:
                checkpoint = torch.load(
                    checkpoint_path, map_location=device, weights_only=weights_only
                )

                if "config" in checkpoint:
                    config = checkpoint["config"]
                    self._merge_config(model_kwargs, config)

                # Load the model weights and infer the config.
                state_dict = checkpoint
                if "model_state_dict" in checkpoint:
                    state_dict = checkpoint["model_state_dict"]
                elif "state_dict" in checkpoint:
                    # This is for backward compatibility with older checkpoints
                    # that used "state_dict" instead of "model_state_dict"
                    state_dict = checkpoint["state_dict"]
                infer_config = getattr(
                    model_class,
                    "infer_config_from_state_dict",
                    lambda _state_dict: {},  # Default to empty dict if not defined
                )
                infer_kwargs = (
                    {"rnn_type": model_kwargs["rnn_type"]}
                    if "rnn_type" in model_kwargs
                    else {}
                )
                self._merge_config(
                    model_kwargs,
                    infer_config(state_dict, **infer_kwargs),
                )

            except Exception as e:
                raise RuntimeError(
                    f"Failed to load checkpoint from {checkpoint_path}: {str(e)}"
                )

        # Filter model_kwargs to only include valid parameters for the model class
        filtered_kwargs = self._filter_model_kwargs(model_class, model_kwargs)

        # Remove None values from filtered_kwargs to avoid passing them to the model constructor
        # This should only happen for parameters that weren't inferred from the checkpoint
        final_kwargs = {k: v for k, v in filtered_kwargs.items() if v is not None}

        # Create the model with the final kwargs
        model = model_class(**final_kwargs)

        # Finally, load the weights.
        if state_dict:
            if state_dict_prefix:
                # If a prefix is provided, filter the state_dict keys
                state_dict = {
                    k[len(state_dict_prefix) :]: v
                    for k, v in state_dict.items()
                    if k.startswith(state_dict_prefix)
                }
            # Load the model weights
            missing, unexpected = model.load_state_dict(
                state_dict, strict=False, assign=True
            )
            if missing or unexpected:
                ez.logger.warning(
                    f"Partial load: missing keys: {missing}, unexpected keys: {unexpected}"
                )

        model.to(device)
        return model
