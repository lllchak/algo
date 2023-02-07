'''
Best model searcher class
'''

from abc import (
    ABC,
    abstractmethod
)

from typing import (
    Any,
    Dict,
    List,
    Sequence,
    Tuple,
    Union,
    Optional,
    Callable
)
from copy import copy

import pandas as pd


class BaseAlgo(ABC):

    _name = "BaseAlgo"

    def __init__(
        self,
        train: pd.DataFrame,
        valid: pd.DataFrame,
        models_dict: Dict[str, Callable]
    ) -> None:

        pass