import logging
import warnings
from typing import (
    List,
    Tuple,
    Dict,
    Union,
    Callable,
    Any
)

import pandas as pd
from sklearn.model_selection import (
    GridSearchCV,
    ParameterGrid
)

from .utils import (
    Models,
    BINARY_PARAMS,
    REG_PARAMS
)
from ..ml import (
    RegressionAlgo,
    ClassificationAlgo
)
from ..features import Preprocessor
from ..task import Task

logging.getLogger().setLevel(logging.INFO)
warnings.filterwarnings("ignore")


class AlgoAutoML:
    """
    Description:
        Auto ML class. Performs basic data preparation, best model searching, 
        ftraining on searched model and predicting the target value

    Methods:
        fit(...) -> None: 
            Running full auto ML pipeline from preparing data to fitting most 
            suitable model with provided data. Returns nothing since it responsible
            for transforming data and finding best model for provided data

        __search_best_model(...) -> Union[ClassficationAlgo, RegressionAlgo]:
            Runs best model searching algorithm using grid search cross-validation
            approach. Returns Algo model base class with respect to the task
            trying to solve
    """

    def __init__(
        self,
        task: Task = None,
        custom_models: List[Union[RegressionAlgo, ClassificationAlgo]] = [],
        only_custom: bool = False
    ) -> None:
        """
        Description:
            AlgoAutoML class constructor. Initializes auto ML pipeline class

        Args:
            task (algo.task.Task) : Solving task object. `name` attribute could be 
                                    `binary` (binary classification) or 
                                    `reg` (regresssion) telling what task we try to solve.
            custom_models (list)  : User provided models. Should have fit(), predict()
                                    methods in their API to match algo.ml.BaseAlgo API.
            only_custom (bool)    : If use only custom models while searching for the
                                    best one flag

        Returns:
            None (only initializes AlgoAutoML object)
        """

        if not task: raise AttributeError("Provide solving task [binary, reg]")
        self.task: Task = task
        self._models: List[Callable] = None
        self.best_estimator_: Callable = None
        self.params_: Dict[str, Any] = None

        if task.name == "binary":
            if not only_custom:
                self._models = Models.BINARY.value + custom_models
            else: self._models = custom_models

        else:
            if not only_custom:
                self._models = Models.REG.value + custom_models
            else: self._models = custom_models

    def fit(
        self,
        data: pd.DataFrame,
        column_roles: Dict[str, Any] = None
    ) -> None:
        """
        Description:
            Main auto ML pipeline runner. Transforms data, searchs for best model,
            saves its state. Throw exceptions if given data or column roles does 
            not match or even exist

        Args:
            data (pd.DataFrame) : Input data in pandas DataFrame format
            column_roles (dict) : Instructions on how to process data 
                                  (what is the target value or which to drop) 

        Returns:
            None (saves best model state after running full auto ML pipeline)
        """

        self.__validate_input(
            data=data, 
            column_roles=column_roles
        )

        prep: Preprocessor = Preprocessor(
            column_roles=column_roles,
            task=self.task
        )

        data = data.drop(columns=column_roles["drop"])
        data = prep.fit_transform(data)

        self.best_estimator_ = self.__search_best_model(data=data, column_roles=column_roles)

    def __search_best_model(
        self,
        data: pd.DataFrame,
        column_roles: Dict[str, Any],
        cv: int = 3
    ) -> Union[ClassificationAlgo, RegressionAlgo]:
        """
        Description:
            Searching best estimator for provided data function

        Args:
            data (pd.DataFrame) : Input data in pandas DataFrame format
            column_roles (dict) : Instructions on how to process data 
                                  (what is the target value or which to drop)
            cv (int)            : Cross validation fold number

        Returns:
            Best ML model with the the respect to the task we are trying to solve
        """

        task_params = self.__define_task_params()
        candidates: List[Callable] = task_params[0]
        param_grid: Dict[str, Any] = task_params[1]
        algo: Callable = task_params[2]
        best_score: float = 0
                                     
        X_train = data.drop(columns=column_roles["target"])
        y_train = data[column_roles["target"]]

        for model in candidates:
            logging.info(
                f" Searching best parameters for {model} [{len(ParameterGrid(param_grid[model]))} fit(-s)]"
            )
            estimator = model()
            searcher = GridSearchCV(
                estimator,
                param_grid=param_grid[model],
                cv=cv,
            )
            searcher.fit(X_train, y_train)
            if searcher.best_score_ > best_score:
                self.best_estimator_ = searcher.best_estimator_
                best_score = searcher.best_score_

        return algo(
            model=self.best_estimator_,
            column_roles=column_roles
        )

    def __define_task_params(
        self
    ) -> Tuple[List[Callable], Dict[str, Any], Callable]:
        """
        Description:
            Function to define task parameters (models to choose from, searching parameters
            and algorithm object)

        Return:
            Task (binary classification or regression) parameters
        """

        if self.task.name == "binary":
            return (
                Models.BINARY.value,
                BINARY_PARAMS,
                ClassificationAlgo
            )
        elif self.task.name == "reg":
            return (
                Models.REG.value,
                REG_PARAMS,
                RegressionAlgo
            )
        else: 
            raise AttributeError(
                "Can only solve two tasks [`binary` (binary classification), `reg` (regression)]"
            )

    def __validate_input(
        self,
        data: pd.DataFrame,
        column_roles: Dict[str, Any]
    ) -> None:
        """
        Funtion to validate input data and its column roles. Check if user provided with
        target variable, if drop columns are in data

        Args:
            data (pd.DataFrame) : Input data in pandas DataFrame format
            column_roles (dict) : Instructions on how to process data 
                                  (what is the target value or which to drop)

        Returns:
            Throws exception if one of the input values do not match the solving task, 
            otherwise returns nothing
        """

        if not column_roles: 
            raise AttributeError("Provide model with data roles (Eg. set target variable)")
        if "target" not in list(column_roles.keys()): 
            raise AttributeError("Provide target variable name from your data")
        if column_roles["drop"]:
            columns: List[str]
            if isinstance(column_roles["drop"], str): columns = [column_roles["drop"]]
            else: columns = column_roles["drop"]

            for col in columns:
                if col not in data.columns:
                    raise KeyError(f'Column "{col}" not found in given data')
