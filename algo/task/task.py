'''
Solving task class
'''

from enum import Enum
from typing import (
    List,
    Dict,
    Callable,
    Optional
)

from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_log_error,
    log_loss,
    auc,

)

class TaskName(Enum): 
    BINARY: str = "binary"
    REG: str = "reg"


class Metrics(Enum):
    DEFAULT_LOSS_BINARY: Callable = log_loss
    DEFAULT_LOSS_REG: Callable = mean_squared_error

    DEFAULT_METRIC_BINARY: Callable = auc
    DEFAUTL_METRIC_REG: Callable = mean_squared_error

    VALID_LOSS_NAME_BINARY: List[Callable] = [log_loss]
    VALID_LOSS_NAME_REG: List[Callable] = [
        mean_squared_error, mean_absolute_error, 
        mean_absolute_percentage_error, mean_squared_log_error
    ]


class Task:
    def __init__(
        self,
        name: str,
        loss: Optional[str] = None,
        metric: Optional[str] = None,
    ) -> None:
        __valid_tasks_names = [tname.value for tname in TaskName] 
        assert name in __valid_tasks_names, \
        f"Invalid task name {name}, choose one of the following: {__valid_tasks_names}"

        self.name = name
        self.loss = self.__get_def_task_loss() if not loss else loss
        self.metric = self.__get_def_task_metric() if not metric else metric

    def __get_def_task_metric(self) -> Callable:
        assert self.name, "Provide task name"

        return Metrics.DEFAULT_METRIC_BINARY \
            if self.name == TaskName.BINARY.value \
            else Metrics.DEFAUTL_METRIC_REG

    def __get_def_task_loss(self) -> Callable:
        assert self.name, "Provide task name"

        return Metrics.DEFAULT_LOSS_BINARY \
            if self.name == TaskName.BINARY.value \
            else Metrics.DEFAULT_LOSS_REG