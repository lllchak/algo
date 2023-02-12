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
