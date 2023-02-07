'''
Best model searcher class
'''

from abc import (
    ABC,
    abstractmethod
)

from typing import (
    Any,
    Dict
)

import numpy as np
import pandas as pd


class BaseAlgo(ABC):

    def __repr__(self) -> str:
        return (
            f"BaseAlgo(\n\ttrain={self._train.shape},\n\tvalid={self._valid.shape},\n\tmodel_params=()\n)"
        )

    def __init__(
        self,
        train: pd.DataFrame,
        test: pd.DataFrame,
        roles: Dict[str, Any] = None
    ) -> None:

        if not roles: raise AttributeError("Provide model with data roles (Eg. set target variable)")
        if not roles["target"]: raise AttributeError("Provide target variable for training")

        self._roles: Dict[str, Any] = roles
        self._train: pd.DataFrame = train
        self._test: pd.DataFrame = test
        self._features: pd.DataFrame = None
        self._target: pd.Series = None
        self.__split_data()

    @abstractmethod
    def fit(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def predict(self) -> np.array:
        raise NotImplementedError

    @abstractmethod
    def fit_predict(self) -> pd.DataFrame:
        raise NotImplementedError

    def __split_data(self) -> None:
        self._features = self._train.drop(columns=self._roles["target"])
        self._target = self._train[self._roles["target"]]
        