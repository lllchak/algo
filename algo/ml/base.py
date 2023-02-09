'''
Best model searcher class
'''

from abc import (
    ABC,
    abstractmethod
)

from typing import (
    Any,
    Tuple,
    Dict
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
        column_roles: Dict[str, Any] = None
    ) -> None:

        if not column_roles: raise AttributeError("Provide model with data roles (Eg. set target variable)")
        if not column_roles["target"]: raise AttributeError("Provide target variable for training")

        self._roles: Dict[str, Any] = column_roles
        self._features = None
        self._target = None

    @abstractmethod
    def fit(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def predict(self) -> np.array:
        raise NotImplementedError

    @abstractmethod
    def fit_predict(self) -> pd.DataFrame:
        raise NotImplementedError

    def _split_data(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        return (
            data.drop(columns=[self._roles["target"]]),
            data[self._roles["target"]]
        )
        