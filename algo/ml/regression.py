import logging
from enum import Enum
from typing import (
    List,
    Dict,
    Any,
    Callable
)

from .base import BaseAlgo

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


class RegressionAlgo(BaseAlgo):

    def __repr__(self) -> str:
        return (
            f"RegressionAlgo(\n\ttrain={self._train.shape},\n\tvalid={self._valid.shape},\n\tmodel_params={self._model_params}\n)"
        )

    def __init__(
        self,
        train: pd.DataFrame,
        test: pd.DataFrame,
        roles: Dict[str, Any] = None,
        model_params: Dict[str, Any] = None,
        model: Callable = LinearRegression
    ) -> None:
        super(RegressionAlgo, self).__init__(train=train, test=test, roles=roles)

        self.__is_fitted: bool = False

        try:
            self.__model = model(model_params)
            self._model_params = model_params
        except:
            logging.warning(f"{model_params} are invalid, using default")
            self.__model = model()

    def fit(self) -> None:
        # self-check
        if not self._features or not self._target:
            raise AttributeError("Input data not splitted")

        # controversial processing, in theory sklearn will throw 
        # it himself if something goes wrong        
        try:
            self.__model.fit(self._features, self._target)
            self.__is_fitted = True
        except:
            raise RuntimeError("Fitting suddenly crashed")

    def predict(self) -> np.array:
        if self.__is_fitted:
            # same thing as in fit method 
            try:
                return self.__model.predict(self._test)
            except:
                raise AttributeError("Invalid data format provided")
        else:
            raise RuntimeError("Can't predict with unfitted model")

    def fit_predict(self) -> np.array:
        self.fit()
        return self.predict(self._test)