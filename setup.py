from setuptools import setup

setup(
    name="algo-auto-ml",
    version="0.1.7",
    description="AutoML library for binary classification and regression tasks",
    url="https://github.com/lllchak/algo",
    author="Pavel Lyulchak",
    author_email="mediumchak@yandex.ru",
    license="MIT",
    packages=[
        "algo",
        "algo/automl",
        "algo/ml",
        "algo/task",
        "algo/features"
    ],
    install_requires=[
        "numpy",
        "pandas",
        "scikit-learn",
        "lightgbm",
        "xgboost"
    ],
    classifiers=[
        "Development Status :: 1 - Planning",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python"
    ]
)