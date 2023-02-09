from enum import Enum
from typing import (
    List,
    Dict,
    Union,
    Callable,
    Any
)

from sklearn.model_selection import train_test_split
from sklearn.linear_model import (
    LinearRegression,
    LogisticRegression
)
from lightgbm import (
    LGBMClassifier,
    LGBMRegressor
)
from xgboost import (
    XGBClassifier,
    XGBRegressor
)

from ..ml import (
    RegressionAlgo,
    ClassificationAlgo
)
from ..features import Preprocessor
from ..task import Task


class Models(Enum):
    CLASSIFICATION = [
        LogisticRegression,
        LGBMClassifier,
        XGBClassifier
    ]

    REGRESSION = [
        LinearRegression,
        LGBMRegressor,
        XGBRegressor
    ]


class AlgoAutoML:
    def __init__(
        self,
        task: Task = None,
        custom_models: List[Union[RegressionAlgo, ClassificationAlgo]] = [],
        only_custom: bool = False
    ) -> None:
        if not task: raise AttributeError("Provide solving task [binary, reg]")
        self.task: Task = task
        if not only_custom:
            self._models: List[Callable] = Models.CLASSIFICATION.value \
                                           if task.name == "binary" \
                                           else Models.REGRESSION.value
            self._models += custom_models
        else: self._models: List[Callable] = custom_models
        self.best_estimator_: Callable = None
        
        