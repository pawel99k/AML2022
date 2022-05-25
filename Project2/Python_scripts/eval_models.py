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
    logistic_model = LogisticRegression(penalty="l2", max_iter=10000)
    logistic_model.fit(X_train, y_train)
    logistic_BA = balanced_accuracy_score(y_true=y_test, y_pred=logistic_model.predict(X_test))
    if verbose:
        print("Logistic model:    ", logistic_BA)

    random_forest_model = RandomForestClassifier(n_jobs=-1)  # do sprawdzenia parametry
    random_forest_model.fit(X_train, y_train)
    random_forest_BA = balanced_accuracy_score(y_true=y_test, y_pred=random_forest_model.predict(X_test))
    if verbose:
        print("RandomForest model:", random_forest_BA)

    adaboost_model = AdaBoostClassifier(n_estimators=100)
    adaboost_model.fit(X_train, y_train)
    adaboost_BA = balanced_accuracy_score(y_true=y_test, y_pred=adaboost_model.predict(X_test))
    if verbose:
        print("AdaBoost model:    ", adaboost_BA)

    lightgbm_model = LGBMClassifier()
    lightgbm_model.fit(X_train, y_train)
    lightgbm_BA = balanced_accuracy_score(y_true=y_test, y_pred=lightgbm_model.predict(X_test))
    if verbose:
        print("LightGBM model:    ", lightgbm_BA)

    xgboost_model = XGBClassifier()  # do sprawdzenia parametry
    xgboost_model.fit(X_train, y_train)
    xgboost_BA = balanced_accuracy_score(y_true=y_test, y_pred=xgboost_model.predict(X_test))
    if verbose:
        print("XGBoost model:     ", xgboost_BA)

    return pd.DataFrame(data={"Classifier": ["LogisticReg", "RandomForest", "AdaBoost", "LightGBM", "XGBoost"],
                       "Balaned Accuracy": [logistic_BA, random_forest_BA, adaboost_BA, lightgbm_BA, xgboost_BA]})
