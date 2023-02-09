"""
Classification task model
"""

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
    model_params=({self._model_params})
)"""
        )

    def __init__(
        self,
        column_roles: Dict[str, Any] = None,
        model_params: Dict[str, Any] = None,
        model: Callable = LogisticRegression
    ) -> None:
        super(ClassificationAlgo, self).__init__(
            column_roles=column_roles,
            model_params=model_params,
            model=model
        )
