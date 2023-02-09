from typing import (
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
            f"""RegressionAlgo(
    model_params=({self._model_params})
)"""
        )

    def __init__(
        self,
        column_roles: Dict[str, Any] = None,
        model_params: Dict[str, Any] = None,
        model: Callable = LinearRegression
    ) -> None:
        super(RegressionAlgo, self).__init__(
            column_roles=column_roles,
            model_params=model_params,
            model=model
        )
