from collections import defaultdict
from copy import copy
from typing import (
    List,
    Dict,
    Callable,
    Optional,
    Any
)

from ..task import Task

import pandas as pd
import numpy as np
from sklearn.preprocessing import (
    StandardScaler,
    LabelEncoder
)


class Preprocessor:
    """
    Description:
        Data transforming class. Defines method for transforming and restore dataset
        variables. `StantardScaler` is used to transform numeric features and
        `LabelEncoder` is used for categorical features. Stores all column -> encoder pairs

    Methods:
        fit(...) -> None:
            Fitting on given variable method. Runs through entire dataset and
            stores transform rules for each variable

        transform(...) -> pd.DataFrame:
            Runs transformation if Preprocessor is fitted. Since we know all the
            rules of transfomation, we can simply apply them in that method and
            get transformed data as the output

        reverse_tranform(...) -> pd.DataFrame:
            Since fit() method stored all col -> encoder pairs we can simply get init variable
            values back
        
        fit_transform(...) -> pd.DataFrame:
            fit() and transform() methods wrapper method. Runs fit-transform pipeline to make data
            transfomation process more convinient

        _get_col_type(...) -> bool:
            _is_cat method helper. Checks if given column is a categorical or object column

        _is_cat(...) -> bool:
            Checks if given column is a categorical column considering that user could provide
            his own list of categorical columns
    """

    def __init__(
        self,
        column_roles: Dict[str, Any],
        task: Task,
        cat_cols: Optional[List[str]] = None
    ) -> None:
        """
        Description:
            Preprocessor class constructor. Initalizes preprocessor object to somehow preprocess
            input data

        Args:
            column_roles (dict) : Instructions on how to process data 
                                  (what is the target value or which to drop)
            task (Task)         : Solving task object. `name` attribute could be 
                                  `binary` (binary classification) or 
                                  `reg` (regresssion) telling what task we try to solve. 
            cat_cols (list)     : List of categorical column provided by user

        Returns:
            None (only initialize Preprocessor object)
        """

        self._col_to_tranformer: Dict[str, Callable] = defaultdict(str)
        self._roles = column_roles
        self.cat_cols = cat_cols
        self._task = task
        self.__cat_cols = []

    def fit(self, data: pd.DataFrame) -> None:
        """
        Description:
            Fitting on given variable method. Runs through entire dataset and
            stores transform rules for each variable

        Args:
            data (pd.DataFrame) : Input data in pandas DataFrame format

        Returns:
            None (only stores encoder for each column in dataset)
        """

        try:
            is_cat: bool = None
            for col in data.columns:
                is_cat = self._is_cat(data, col)

                transformer = LabelEncoder() \
                    if is_cat \
                    else StandardScaler()
                
                if is_cat:
                    data[col] = data[col].fillna("NULL")
                    transformer.fit(data[col])
                    self.__cat_cols.append(col)
                else:
                    if self._roles["target"] != col:
                        data[col] = data[col].fillna(data[col].median())
                        transformer.fit(np.array(data[col]).reshape(-1, 1))
                self._col_to_tranformer[col] = transformer

        except: raise ValueError("Dataset should in pandas format")

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Description:
            Runs transformation if Preprocessor is fitted. Since we know all the
            rules of transfomation, we can simply apply them in that method and
            get transformed data as the output

        Args:
            data (pd.DataFrame) : Input data in pandas DataFrame format

        Returns:
            Transformed data in pandas DataFrame format
        """

        _tmp: pd.DataFrame = copy(data)

        try:
            transformed_dict: Dict[str, List[Any]] = {col: None for col in _tmp.columns}

            is_cat: bool = None
            for col in _tmp.columns:
                if col in self.__cat_cols:
                    data[col] = data[col].fillna("NULL")
                    transformed_dict[col] = (
                        self._col_to_tranformer[col].transform(data[col])
                    )
                else: 
                    data[col] = data[col].fillna(data[col].median())
                    if self._roles["target"] != col:
                        transformed_dict[col] = (
                            self._col_to_tranformer[col].transform(np.array(data[col]).reshape(-1, 1)).ravel()
                        )
                    else: transformed_dict[col] = data[col].ravel()

            return pd.DataFrame(transformed_dict)

        except: raise ValueError("Dataset should be in pandas format")

    def reverse_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Description:
            Since fit() method stored all col -> encoder pairs we can simply get init variable
            values back

        Args:
            data (pd.DataFrame) : Input data in pandas DataFrame format

        Returns:
            Restored data in pandas DataFrame format
        """

        _tmp: pd.DataFrame = copy(data)

        try:
            transformed_dict: Dict[str, List[Any]] = {col: None for col in _tmp.columns}

            for col in _tmp.columns:
                if col in self.__cat_cols:
                    transformed_dict[col] = (
                        self._col_to_tranformer[col].inverse_transform(data[col])
                    )
                else: 
                    if self._roles["target"] != col:
                        transformed_dict[col] = (
                            self._col_to_tranformer[col].inverse_transform(np.array(data[col]).reshape(-1, 1)).ravel()
                        )
                    else: transformed_dict[col] = data[col].ravel()

            return pd.DataFrame(transformed_dict)

        except: raise ValueError("Dataset should be in pandas format")

    def fit_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Description:
            fit() and transform() methods wrapper method. Runs fit-transform pipeline to make data
            transfomation process more convinient

        Args:
            data (pd.DataFrame) : Input data in pandas DataFrame format

        Retuns:
            Transformed data in pandas DataFrame format
        """

        self.fit(data)
        return self.transform(data)

    def _get_col_type(self, data: pd.Series) -> bool:
        """
        Description:
            _is_cat method helper. Checks if given column is a categorical or object column

        Args:
            data (pd.Series) : Column to check if categorical

        Returns:
            If column is categorical or not flag
        """

        return data.dtype.name in ["category", "object"]

    def _is_cat(self, data: pd.DataFrame, col: str) -> bool:
        """
        Description:
            Checks if given column is a categorical column considering that user could provide
            his own list of categorical columns

        Args:
            data (pd.DataFrame) : Input data in pandas DataFrame format
            col (str)           : Column to check name

        Returns:
            If column is categorical or not flag  
        """

        is_cat: bool = None
        if not self.cat_cols:
            is_cat = self._get_col_type(data[col])
        else: is_cat = col in self.cat_cols or self._get_col_type(data[col])

        return is_cat
