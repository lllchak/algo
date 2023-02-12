"""
Classification task model
"""

import logging
from typing import (
    List,
    Dict,
    Any,
    Callable
)

from .base import BaseAlgo

from sklearn.linear_model import LogisticRegression


class ClassificationAlgo(BaseAlgo):
    """
    Classification task model wrapper class. Since have similar API for fit(),
    predict() and fit_predict() methods with regression task model, full API 
    is defined in BaseAlgo class [./base.py].
    
    Here simply inherits full BaseAlgo API. To see fit(), predict() and etc. definitions
    and docs see ./base.py -> BaseAlgo class
    """

    def __repr__(self) -> str:
        return (
            f"""ClassificationAlgo(
    estimator=({self._model}),
    column_roles=({self._roles})
)"""
        )

    def __init__(
        self,
        model: Callable,
        column_roles: Dict[str, Any] = None,
    ) -> None:
        if not model: 
            logging.warning(f" Best model not found, using default ({LogisticRegression})")
            model = LogisticRegression()

        super(ClassificationAlgo, self).__init__(
            column_roles=column_roles,
            model=model
        )
