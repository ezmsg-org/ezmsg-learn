from enum import Enum
from dataclasses import dataclass, field

from ezmsg.util.messages.axisarray import AxisArray
import sklearn.linear_model

# from sklearn.neural_network import MLPClassifier
import river.linear_model


class AdaptiveLinearRegressor(str, Enum):
    LINEAR = "linear"
    LOGISTIC = "logistic"
    SGD = "sgd"
    PAR = "par"  # passive-aggressive
    # MLP = "mlp"


class LinearRegressor(str, Enum):
    LINEAR = "linear"
    RIDGE = "ridge"


REGRESSORS = {
    AdaptiveLinearRegressor.LINEAR: river.linear_model.LinearRegression,
    AdaptiveLinearRegressor.LOGISTIC: river.linear_model.LogisticRegression,
    AdaptiveLinearRegressor.SGD: sklearn.linear_model.SGDRegressor,
    AdaptiveLinearRegressor.PAR: sklearn.linear_model.PassiveAggressiveRegressor,
    # LinearRegressors.MLP: MLPClassifier,
    LinearRegressor.LINEAR: sklearn.linear_model.LinearRegression,
    LinearRegressor.RIDGE: sklearn.linear_model.Ridge,
}


@dataclass
class ClassifierMessage(AxisArray):
    labels: list[str] = field(default_factory=list)
