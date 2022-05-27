from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import balanced_accuracy_score
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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


def get_models_ba(X_train, X_test, y_train, y_test,logistic_args={},RF_args={},AdaBoost_args={},LGBM_args={},XDB_args={}, verbose=True):
    # TODO dodać jakiś argument/argumenty pozwalające na dobieranie hiperparametrów modeli
    # najprościej chyba będzie to zrobić przy pomocy słowników

    # to o czym rozmawialiśmy
    X_test = X_test.copy()
    X_test = X_test.loc[:, X_train.columns]

    models = (LogisticRegression(**logistic_args), RandomForestClassifier(**RF_args), AdaBoostClassifier(**AdaBoost_args), LGBMClassifier(**LGBM_args), XGBClassifier(**XDB_args))
    names = []
    BA_scores = []
    for model in models:
        names.append(model.__class__.__name__)
        model.fit(X_train, y_train)
        BA_scores.append(balanced_accuracy_score(y_true=y_test, y_pred=model.predict(X_test)))
        if verbose:
            print(names[-1], round(BA_scores[-1],4))

    return pd.DataFrame(data={"Classifier": names, "BA score": BA_scores})
def standarize(X_train,X_test):
    ss=StandardScaler()
    X_train_sc=pd.DataFrame(ss.fit_transform(X_train), index=X_train.index,columns=X_train.columns)
    X_test_sc=pd.DataFrame(ss.transform(X_test), index=X_test.index, columns=X_test.columns)
    return X_train_sc,X_test_sc

def delete_corr(X,thresh=0.98,verbose=False):
        X=X.copy()
        cor_matrix = X.corr().abs()
        if verbose:
            plt.figure(figsize=(12,12))
            sns.heatmap(cor_matrix,vmin=0.5,vmax=1)
            plt.show()
        upper_tri = cor_matrix.where(np.triu(np.ones(cor_matrix.shape),k=1).astype(bool))
        to_drop = [column for column in upper_tri.columns if any(upper_tri[column] >= thresh)]
        X_cleaned=X.drop(columns=to_drop)
        return X_cleaned