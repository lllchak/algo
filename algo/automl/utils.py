from enum import Enum

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

"""
Default models to choose from. User can also
provide custom models in AlgoAutoML constructor
as a list of class objects.
"""
class Models(Enum):
    BINARY = [
        LogisticRegression,
        LGBMClassifier,
        XGBClassifier
    ]

    REG = [
        LinearRegression,
        LGBMRegressor,
        XGBRegressor
    ]


"""
Classfication task models parameters grids.
To select best model for each task and
dataset. AlgoAutoML class search best
parameters combination using GridSearchCV
"""
BINARY_PARAMS = {
    LogisticRegression: {
        "penalty": ["l2", None],
        "C": [0.1, 0.3, 0.5],
    },

    LGBMClassifier: {
        "max_depth": [5, 7, 13],
        "learning_rate": [0.1, 5e-2],
        "n_estimators": [100, 300, 1000],
        "n_jobs": [-1]
    },

    XGBClassifier: {
        "max_depth": [5, 7, 13],
        "learning_rate": [0.1, 5e-2],
        "n_estimators": [100, 300, 1000],
        "n_jobs": [-1]
    }
}

"""
Regression task models parameters grids.
To select best model for each task and
dataset. AlgoAutoML class search best
parameters combination using GridSearchCV
"""
REG_PARAMS = {
    LinearRegression: {
        "n_jobs": [-1]
    },

    LGBMRegressor:{    
        "max_depth": [5, 7, 13],
        "learning_rate": [0.1, 5e-2],
        "n_estimators": [100, 300, 1000],
        "n_jobs": [-1]
    },

    XGBRegressor: {
        "max_depth": [5, 7, 13],
        "learning_rate": [0.1, 5e-2],
        "n_estimators": [100, 300, 1000],
        "n_jobs": [-1]
    }
}
