'''
Best model searcher class
'''

from abc import (
    ABC,
    abstractmethod
)
import logging

from typing import (
    Any,
    Tuple,
    Dict,
    Callable
)

import numpy as np
import pandas as pd


class BaseAlgo(ABC):

    def __repr__(self) -> str:
        return (
            f"""BaseAlgo(
    model_params=()
)"""
        )

    def __init__(
        self,
        column_roles: Dict[str, Any] = None,
        model_params: Dict[str, Any] = None,
        model: Callable = None
    ) -> None:

        if not column_roles: raise AttributeError("Provide model with data roles (Eg. set target variable)")
        if not column_roles["target"]: raise AttributeError("Provide target variable for training")

        self._roles: Dict[str, Any] = column_roles
        self._features = None
        self._target = None

        self.__is_fitted: bool = False

        try:
            self.__model = model(model_params)
            self._model_params = model_params
        except:
            logging.warning(f" Model params are not provided, using default")
            self.__model = model()
            self._model_params = None


    def fit(self, train_data: pd.DataFrame) -> None:
        self._features, self._target = (
            self._split_data(train_data)
        )

        # self-check
        if not len(self._features) or not len(self._target):
            raise AttributeError("Input data not splitted")

        # controversial processing, in theory sklearn will throw 
        # it himself if something goes wrong        
        try:
            # TODO: check target dimensions (dould be 2d)
            self.__model.fit(self._features, self._target)
            self.__is_fitted = True
        except:
            raise RuntimeError("Fitting suddenly crashed")

    def predict(self, test_data) -> np.array:
        if self.__is_fitted:
            # same thing as in fit method 
            try:
                return self.__model.predict(test_data)
            except:
                raise AttributeError("Invalid data format provided")
        else:
            raise RuntimeError("Can't predict with unfitted model")

    def fit_predict(
        self,
        train_data: pd.DataFrame
    ) -> np.array:
        self.fit(train_data=train_data)
        return self.predict(test_data=train_data.drop(columns=self._roles["target"]))

    def _split_data(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        return (
            data.drop(columns=[self._roles["target"]]),
            data[self._roles["target"]]
        )
        