"""
Regression task model
"""

import logging
from typing import (
    Dict,
    Any,
    Callable
)

from .base import BaseAlgo

from sklearn.linear_model import LinearRegression


class RegressionAlgo(BaseAlgo):

    def __repr__(self) -> str:
        return (
            f"""RegressionAlgo(
    estimator=({self._model}),
    column_roles=({self._roles})
)"""
        )

    def __init__(
        self,
        column_roles: Dict[str, Any] = None,
        model: Callable = LinearRegression()
    ) -> None:
        if not model: 
            logging.warning(f" Best model not found, using default ({LinearRegression})")
            model = LinearRegression()

        super(RegressionAlgo, self).__init__(
            column_roles=column_roles,
            model=model
        )
