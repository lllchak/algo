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
        try:
            is_cat: bool = None
            for col in data.columns:
                is_cat = self._is_cat(data, col)

                transformer = LabelEncoder() \
                    if is_cat \
                    else StandardScaler()
                
                if is_cat:
                    data[col].fillna("NULL")
                    transformer.fit(data[col])
                else: 
                    data[col].fillna(data[col].median())
                    transformer.fit(np.array(data[col]).reshape(-1, 1))
                self._col_to_tranformer[col] = transformer

        except: raise ValueError("Dataset should in pandas format")

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        tmp: pd.DataFrame = copy(data)

        try:
            transformed_dict: Dict[str, List[Any]] = {col: None for col in tmp.columns}

            is_cat: bool = None
            for col in tmp.columns:
                is_cat = self._is_cat(data, col)

                if is_cat:
                    data[col].fillna("NULL")
                    transformed_dict[col] = (
                        self._col_to_tranformer[col].transform(data[col])
                    )
                else: 
                    data[col].fillna(data[col].median())
                    transformed_dict[col] = (
                        self._col_to_tranformer[col].transform(np.array(data[col]).reshape(-1, 1)).ravel()
                    )

            return pd.DataFrame(transformed_dict)

        except: raise ValueError("Dataset should be in pandas format")

    def fit_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        self.fit(data)
        return self.transform(data)

    def _get_col_type(self, data: pd.Series) -> List[str]:
        return data.dtype.name in ["category", "object"]

    def _is_cat(self, data: pd.DataFrame, col: str) -> bool:
        is_cat: bool = None
        if not self.cat_cols:
            is_cat = self._get_col_type(data[col])
        else: is_cat = col in self.cat_cols or self._get_col_type(data[col])

        return is_cat