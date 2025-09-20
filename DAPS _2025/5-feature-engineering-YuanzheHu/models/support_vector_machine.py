import numpy as np
import pandas as pd
from sklearn.svm import SVC


def train(X_train, y_train):
    """Train a support vector machine classifier on the training data."""
    if isinstance(X_train, pd.DataFrame):
        X_train_tmp = X_train.values
    elif isinstance(X_train, np.ndarray):
        X_train_tmp = X_train
    else:
        raise ValueError("X_train must be a pandas dataframe or numpy array")

    if isinstance(y_train, pd.DataFrame):
        y_train_tmp = y_train.values.ravel()
    elif isinstance(y_train, np.ndarray):
        y_train_tmp = np.ravel(y_train)
    else:
        raise ValueError("y_train must be a pandas dataframe or numpy array")

    # Train the Support Vector Machine model
    model = SVC(probability=True).fit(X_train_tmp, y_train_tmp)

    return model


def evaluate(model, X_train, y_train, X_test, y_test):
    """Evaluate the performance of a model on the training and test sets."""
    if isinstance(X_train, pd.DataFrame):
        X_train_tmp = X_train.values
    elif isinstance(X_train, np.ndarray):
        X_train_tmp = X_train
    else:
        raise ValueError("X_train must be a pandas dataframe or numpy array")

    if isinstance(X_test, pd.DataFrame):
        X_test_tmp = X_test.values
    elif isinstance(X_test, np.ndarray):
        X_test_tmp = X_test
    else:
        raise ValueError("X_test must be a pandas dataframe or numpy array")

    if isinstance(y_train, pd.DataFrame):
        y_train_tmp = y_train.values.ravel()
    elif isinstance(y_train, np.ndarray):
        y_train_tmp = np.ravel(y_train)
    else:
        raise ValueError("y_train must be a pandas dataframe or numpy array")

    if isinstance(y_test, pd.DataFrame):
        y_test_tmp = y_test.values.ravel()
    elif isinstance(y_test, np.ndarray):
        y_test_tmp = np.ravel(y_test)
    else:
        raise ValueError("y_test must be a pandas dataframe or numpy array")

    print(f"Score on training set: {100*model.score(X_train_tmp, y_train_tmp):.2f}%")
    print(f"Score on test set: {100*model.score(X_test_tmp, y_test_tmp):.2f}%")
