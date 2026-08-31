import typing
from dataclasses import dataclass, field
from enum import Enum

from ezmsg.util.messages.axisarray import AxisArray

from ._optional import missing_extra

# from sklearn.neural_network import MLPClassifier


class RegressorType(str, Enum):
    ADAPTIVE = "adaptive"
    STATIC = "static"


class AdaptiveLinearRegressor(str, Enum):
    LINEAR = "linear"
    LOGISTIC = "logistic"
    SGD = "sgd"
    PAR = "par"  # passive-aggressive
    # MLP = "mlp"


class StaticLinearRegressor(str, Enum):
    LINEAR = "linear"
    RIDGE = "ridge"


# The registries below are built on demand so that the enums and
# :class:`ClassifierMessage` -- which need nothing beyond ezmsg -- stay importable
# without the ``sklearn`` extra installed.
def _adaptive_regressors() -> dict:
    try:
        import river.linear_model
        import sklearn.linear_model
    except ImportError as exc:
        raise missing_extra("sklearn", __name__) from exc

    return {
        AdaptiveLinearRegressor.LINEAR: river.linear_model.LinearRegression,
        AdaptiveLinearRegressor.LOGISTIC: river.linear_model.LogisticRegression,
        AdaptiveLinearRegressor.SGD: sklearn.linear_model.SGDRegressor,
        AdaptiveLinearRegressor.PAR: sklearn.linear_model.PassiveAggressiveRegressor,
        # AdaptiveLinearRegressor.MLP: MLPClassifier,
    }


def _static_regressors() -> dict:
    try:
        import sklearn.linear_model
    except ImportError as exc:
        raise missing_extra("sklearn", __name__) from exc

    return {
        StaticLinearRegressor.LINEAR: sklearn.linear_model.LinearRegression,
        StaticLinearRegressor.RIDGE: sklearn.linear_model.Ridge,
    }


def __getattr__(name: str) -> dict:
    """Resolve the ``*_REGRESSORS`` registries lazily (:pep:`562`)."""
    if name == "ADAPTIVE_REGRESSORS":
        return _adaptive_regressors()
    if name == "STATIC_REGRESSORS":
        return _static_regressors()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# Function to get a regressor by type and name
def get_regressor(
    regressor_type: typing.Union[RegressorType, str],
    regressor_name: typing.Union[AdaptiveLinearRegressor, StaticLinearRegressor, str],
):
    if isinstance(regressor_type, str):
        regressor_type = RegressorType(regressor_type)

    if regressor_type == RegressorType.ADAPTIVE:
        if isinstance(regressor_name, str):
            regressor_name = AdaptiveLinearRegressor(regressor_name)
        return _adaptive_regressors()[regressor_name]
    elif regressor_type == RegressorType.STATIC:
        if isinstance(regressor_name, str):
            regressor_name = StaticLinearRegressor(regressor_name)
        return _static_regressors()[regressor_name]
    else:
        raise ValueError(f"Unknown regressor type: {regressor_type}")


@dataclass
class ClassifierMessage(AxisArray):
    labels: list[str] = field(default_factory=list)
