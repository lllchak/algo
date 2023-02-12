# Algo
Algo is as AutoML framework. It provides automatic model creation for the following tasks:
- Binary classification
- Regression

Current version of the package handles pandas DataFrame format datasets that have independent samples in each row.

**Note**: Package using `Scikit-learn` models and preprocessors

# Table of contents
- [Algo installation and uninstallation](#installation)
- [Quick tour](#quick-tour)
- [License](#license)

# Installation
To `install` Algo library on your local machine, run following
```
pip install -U algo-auto-ml
```
**Note**: It is better to set up `virtual environment` before running installation. To do so, run
```
python -m venv <virtual_env_name>

source virtual_env_name/bin/activate
```

To `uninstall` Algo simply run
```
pip uninstall algo-auto-ml -y
```
[Back on top](#table-of-contents)

# Quick tour
Let's solve popular Kaggle competition `Titanic - Machine Learning from Disaster`. To do this simply create `AlgoAutoML` class and provide it with the solving task [`binary` (binary classification) or `reg` (regression)]
```python []
import pandas as pd

from algo.automl import AlgoAutoML
from algo.task import Task

data_train = pd.read_csv("train.csv")
data_test = pd.read_csv("test.csv")

automl = AlgoAutoML(
    task=Task(name="binary")
)

automl.fit(
    data=data_train,
    column_roles={"target": "Survived", "drop": "PassengerId"}
)

preds = automl.predict(data=data_test)

result = pd.DataFrame(
    {
        "PassengerId": list(range(1, len(data_test) + 1)),
        "Survived": (preds > 0.5)*1
    }
)

result.to_csv("submit.csv", index=False)
```

After `AlgoAutoML` algorithm is fitted, you can run `predict` method to get predictions

Also, you can build your own models with `ClassificationAlgo` and `RegressionAlgo` objects like this
```python []
from algo.ml import ClassificationAlgo
from algo.features import Preprocessor
from algo.task import Task

import pandas as pd
from sklearn.linear_model import LogisticRegression

data_train = pd.read_csv("train.csv")
data_test = pd.read_csv("test.csv")
task = Task(name="binary")

column_roles = {"target": "Survived", "drop": "PassengerId"}
prep = Preprocessor(
    column_roles=column_roles,
    task=task
)

transformed_train_data = prep.fit_transform(data_train)
transformed_test_data = prep.fit_transform(data_test)

classification_model = ClassificationAlgo(
    model=LogisticRegression(),
    column_roles=column_roles
)

classification_model.fit(train_data=transformed_train_data)

preds = classification_model.predict(test_data=transformed_test_data)

result = pd.DataFrame(
    {
        "PassengerId": list(range(1, len(data_test) + 1)),
        "Survived": (preds > 0.5)*1
    }
)

result.to_csv("submit.csv", index=False)
```
[Back on top](#table-of-contents)

# License
This project is licenced under the MIT License. See [LICENCE](./LICENSE) for more details
[Back on top](#table-of-contents)
