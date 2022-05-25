from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import balanced_accuracy_score
from sklearn.feature_selection import VarianceThreshold
import pandas as pd


def get_data(remove_constant=True):
    X_artificial = pd.read_csv("../data/artificial/artificial_train.data", header=None, delim_whitespace=True)
    y_artificial = pd.read_csv("../data/artificial/artificial_train.labels", header=None, delim_whitespace=True)
    X_digits = pd.read_csv("../data/digits/digits_train.data", header=None, delim_whitespace=True)
    y_digits = pd.read_csv("../data/digits/digits_train.labels", header=None, delim_whitespace=True)
    if remove_constant:
        VT_art = VarianceThreshold(threshold=0)
        VT_art.fit(X_artificial)
        X_artificial = X_artificial.iloc[:, VT_art.get_support(indices=True).tolist()]

        VT_dig = VarianceThreshold(threshold=0)
        VT_dig.fit(X_digits)
        X_digits = X_digits.iloc[:, VT_dig.get_support(indices=True).tolist()]

    y_artificial = ((y_artificial + 1) / 2).to_numpy().ravel()
    y_digits = ((y_digits + 1) / 2).to_numpy().ravel()

    return X_artificial, y_artificial, X_digits, y_digits


def get_models_ba(X_train, X_test, y_train, y_test, verbose=True):
    # TODO dodać jakiś argument/argumenty pozwalające na dobieranie hiperparametrów modeli
    # najprościej chyba będzie to zrobić przy pomocy słowników
    models = (LogisticRegression(), RandomForestClassifier(), AdaBoostClassifier(), LGBMClassifier(), XGBClassifier())
    names = []
    BA_scores = []
    for model in models:
        names.append(model.__class__.__name__)
        model.fit(X_train, y_train)
        BA_scores.append(balanced_accuracy_score(y_true=y_test, y_pred=model.predict(X_test)))
        if verbose:
            print(names[-1], BA_scores[-1])

    return pd.DataFrame(data={"Classifier": names, "BA score": BA_scores})
