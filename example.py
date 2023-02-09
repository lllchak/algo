from algo.task import Task
from algo.ml import (
    BaseAlgo,
    RegressionAlgo
)

import pandas as pd
from sklearn.datasets import load_diabetes


if __name__ == "__main__":
    task = Task(name="binary")
    print(task._loss, task._name)

    data = pd.read_csv("test.csv")

    algo = RegressionAlgo(column_roles={"target": "target"})
    algo.fit_predict(data, data[["row1", "row2"]])

    print(algo)