"""
Base ml model class
"""

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
    """
    Model base class. Since using sklearn, lightgbm and xgboost
    which have save API for fitting and inference for modeling
    fit(), predict() and fit_predict() methods defined in base class
    """

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
        """
        Description:
            Initialize base model with provided model 
            (depends on solving task (regression, classification))

        Args:
            column_roles (Dict[str, Any]) : Columns processing instructions (drop, target, etc.)
            model_params (Dict[str, Any]) : Model parameters (regularization constant, etc.)
            model        (Callable)       : Machine learning model class

        Returns:
            None (only initialize base model)
        """

        if not model: raise AttributeError("No model class provided")
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
        """
        Description:
            Model fitting method (simply calls out-of-the-box implementations)
        
        Args:
            train_data (pd.DataFrame) : Training data in pandas dataframe format

        Returns:
            None (fits ml model)
        """

        self._features, self._target = (
            self._split_data(train_data)
        )

        # self-check
        if not len(self._features) or not len(self._target):
            raise AttributeError("Input data not splitted")

        # controversial processing, in theory sklearn 
        # (or lightgbm, xgboost) will throw 
        # it himself if something goes wrong        
        try:
            # TODO: check target dimensions (dould be 2d)
            self.__model.fit(self._features, self._target)
            self.__is_fitted = True
        except: raise RuntimeError("Fitting suddenly crashed")

    def predict(self, test_data: pd.DataFrame) -> np.array:
        """
        Description:
            Predict values based on given dataset

        Args:
            test_data (pd.DataFrame): Data to predict values to in pandas dataframe format

        Returns:
            Predicted values as numpy array
        """

        if self.__is_fitted:
            # as in fit method,
            # controversial processing, in theory sklearn 
            # (or lightgbm, xgboost) will throw 
            # it himself if something goes wrong 
            try: return self.__model.predict(test_data)
            except: raise AttributeError("Invalid data format provided")
        else: raise RuntimeError("Can't predict with unfitted model")

    def fit_predict(
        self,
        train_data: pd.DataFrame
    ) -> np.array:
        """
        Description:
            Wrapper for fit() and predict() methods (simply calls both alternately)

        Args:
            train_data (pd.DataFrame) : Training data in pandas dataframe format

        Returns:
            Predicted values as numpy array
        """

        self.fit(train_data=train_data)
        return self.predict(test_data=train_data.drop(columns=self._roles["target"]))

    def _split_data(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Description:
            Splits target and features columns

        Args:
            data (pd.DataFrame): Dataset to split

        Returns:
            Tuple with dataset features and target
        """

        return (
            data.drop(columns=[self._roles["target"]]),
            data[self._roles["target"]]
        )
        