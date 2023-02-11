from collections import defaultdict
from enum import Enum
from typing import (
    List,
    Dict,
    Union,
    Callable,
    Any
)

from sklearn.model_selection import GridSearchCV
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
import pandas as pd

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


BINARY_PARAMS = {
    LogisticRegression: {
        'penalty': ['l2', 'l1', None],
        'C': [0.1, 0.3, 0.5],
        'solver': ['liblinear']
    },

    LGBMClassifier: {
        'max_depth': [5, 13, -1],
        'learning_rate': [0.1, 5e-2],
        'n_estimators': [100, 300, 1000],
        'n_jobs': [-1]
    },

    XGBClassifier: {
        'max_depth': [5, 13, -1],
        'learning_rate': [0.1, 5e-2],
        'n_estimators': [100, 300, 1000],
        'n_jobs': [-1]
    }
}


class AlgoAutoML:
    def __init__(
        self,
        task: Task = None,
        custom_models: List[Union[RegressionAlgo, ClassificationAlgo]] = [],
        only_custom: bool = False
    ) -> None:
        if not task: raise AttributeError("Provide solving task [binary, reg]")
        self.task: Task = task
        self._models: List[Callable] = None
        self._best_estimator: Callable = None

        if task.name == "binary":
            if not only_custom:
                self._models = Models.CLASSIFICATION.value + custom_models
            else: self._models = custom_models

        else:
            if not only_custom:
                self._models = Models.REGRESSION.value + custom_models
            else: self._models = custom_models

    def fit(
        self,
        data: pd.DataFrame,
        column_roles: Dict[str, Any]
    ) -> None:
        self.__search_best_model(data=data, column_roles=column_roles)

    def __costruct_models(
        self,
        column_roles: Dict[str, Any]
    ) -> List[Callable]:
        if self.task.name == "binary":
            return [
                ClassificationAlgo(
                    model=model, column_roles=column_roles
                ) for model in self._models
            ]
        else:
            return [
                RegressionAlgo(
                    model=model,
                    column_roles=column_roles
                ) for model in self._models
            ]

    def __search_best_model(
        self,
        data: pd.DataFrame,
        column_roles: Dict[str, Any],
        cv: int = 3
    ) -> Callable:
        prep: Preprocessor = Preprocessor()
        candidates: Dict[float, Callable] = defaultdict(float)
        print(data.isnull().sum())

        data = prep.fit_transform(data)
        print(data.isnull().sum())
        data = data.drop(columns=column_roles["drop"])
        X_train = data.drop(columns=column_roles["target"])
        y_train = data[column_roles["target"]]

        if self.task.name == "binary":
            for model in Models.CLASSIFICATION.value:
                estimator = model()
                searcher = GridSearchCV(
                    estimator, 
                    param_grid=BINARY_PARAMS[model],
                    cv=cv
                )
                searcher.fit(X_train, y_train)
                candidates[searcher.best_score_] = searcher.best_estimator_

        print(candidates)


