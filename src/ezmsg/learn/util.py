from enum import Enum
from dataclasses import dataclass, field
import typing

from ezmsg.util.messages.axisarray import AxisArray
import sklearn.linear_model
import river.linear_model
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


ADAPTIVE_REGRESSORS = {
    AdaptiveLinearRegressor.LINEAR: river.linear_model.LinearRegression,
    AdaptiveLinearRegressor.LOGISTIC: river.linear_model.LogisticRegression,
    AdaptiveLinearRegressor.SGD: sklearn.linear_model.SGDRegressor,
    AdaptiveLinearRegressor.PAR: sklearn.linear_model.PassiveAggressiveRegressor,
    # AdaptiveLinearRegressor.MLP: MLPClassifier,
}


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
        return ADAPTIVE_REGRESSORS[regressor_name]
    elif regressor_type == RegressorType.STATIC:
        if isinstance(regressor_name, str):
            regressor_name = StaticLinearRegressor(regressor_name)
        return STATIC_REGRESSORS[regressor_name]
    else:
        raise ValueError(f"Unknown regressor type: {regressor_type}")


STATIC_REGRESSORS = {
    StaticLinearRegressor.LINEAR: sklearn.linear_model.LinearRegression,
    StaticLinearRegressor.RIDGE: sklearn.linear_model.Ridge,
}


@dataclass
class ClassifierMessage(AxisArray):
    labels: list[str] = field(default_factory=list)
