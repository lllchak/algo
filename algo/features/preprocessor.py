from collections import defaultdict
from copy import copy
from typing import (
    List,
    Dict,
    Callable,
    Any
)

import pandas as pd
import numpy as np
from sklearn.preprocessing import (
    StandardScaler,
    LabelEncoder
)


class Preprocessor:
    def __init__(
        self,
        cat_cols: List[str] = None
    ) -> None:
        self._col_to_tranformer: Dict[str, Callable] = defaultdict(str)
        self.cat_cols = cat_cols

    def fit(self, data: pd.DataFrame) -> None:
        is_cat: bool = None
        try:
            for col in data.columns:
                if not self.cat_cols:
                    is_cat = self._get_col_type(data[col])
                else: is_cat = col in self.cat_cols or self._get_col_type(data[col])

                transformer = LabelEncoder() \
                    if is_cat \
                    else StandardScaler()
                
                if is_cat: transformer.fit(data[col])
                else: transformer.fit(np.array(data[col]).reshape(-1, 1))
                self._col_to_tranformer[col] = transformer

        except: raise ValueError("Dataset should in pandas format")

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        tmp: pd.DataFrame = copy(data)

        try:
            transformed_dict: Dict[str, List[Any]] = {
                self._col_to_tranformer[col] for col in tmp.columns
            }
        except: raise ValueError("Dataset should be in pandas format")


        return pd.DataFrame(
            {
                self._col_to_tranformer[col] for col in tmp.columns
            }
        )

    def _get_col_type(self, data: pd.Series) -> List[str]:
        return data.dtype.name in ["category", "object"]