from setuptools import setup

setup(
    name="algo",
    version="0.1.0",
    description="AutoML library for binary classification and regression tasks",
    url="https://github.com/lllchak/algo",
    author="Pavel Lyulchak",
    author_email="mediumchak@yandex.ru",
    license="MIT",
    packages=["algo"],
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