from algo.task import Task
from algo.ml import (
    BaseAlgo,
    RegressionAlgo,
    ClassificationAlgo
)
from algo.features import Preprocessor
from algo.automl import AlgoAutoML

import pandas as pd
from sklearn.datasets import load_diabetes


if __name__ == "__main__":
    task = Task(name="binary")
    print(task.loss, task.name)

    data = pd.read_csv("../train.csv")

    # algo = RegressionAlgo(column_roles={"target": "target"})
    # pred = algo.fit_predict(data)

    # print(algo)
    # print(pred)

    # processor = Preprocessor(cat_cols=["Survived"])
    # processor.fit(data)
    # print(processor.fit_transform(data))

    automl = AlgoAutoML(Task(name="binary"))
    automl.fit(data, column_roles={"target": "Survived", "drop": "Name"})

