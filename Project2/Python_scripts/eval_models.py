from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import balanced_accuracy_score
from sklearn.feature_selection import mutual_info_classif, SelectKBest, SelectFromModel
from sklearn.feature_selection import VarianceThreshold, chi2, f_classif
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import GridSearchCV
from statsmodels.stats.outliers_influence import variance_inflation_factor
import pandas as pd
import numpy as np
import json
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

def get_train_test(set_name):
    if set_name=="artificial":
        path="../data/artificial/"
    elif set_name=="digits":
        path="../data/digits/"
    else:
        print("sth went wrong")
        return
    
    X_train = pd.read_csv(path+"X_train.csv",index_col=0)
    X_train.columns= X_train.columns.astype(np.int64)
    X_test = pd.read_csv(path+"X_test.csv",index_col=0)
    X_test.columns= X_test.columns.astype(np.int64)
    y_train= np.genfromtxt(path+"y_train.csv")
    y_test =np.genfromtxt(path+"y_test.csv")
    return X_train,X_test,y_train,y_test


def get_models_ba(X_train, X_test, y_train, y_test, n_estimators=150, max_iter=1000, logistic_args={}, RF_args={},
                  AdaBoost_args={}, LGBM_args={}, XDB_args={}, verbose=True):

    # to o czym rozmawialiśmy
    X_test = X_test.copy()
    X_test = X_test.loc[:, X_train.columns]

    models = (LogisticRegression(max_iter=max_iter, **logistic_args),
              RandomForestClassifier(n_estimators=n_estimators, **RF_args),
              AdaBoostClassifier(n_estimators=n_estimators, **AdaBoost_args),
              LGBMClassifier(n_estimators=n_estimators, **LGBM_args),
              XGBClassifier(n_estimators=n_estimators, **XDB_args))
    names = []
    BA_scores = []
    for model in models:
        names.append(model.__class__.__name__)
        model.fit(X_train, y_train)
        BA_scores.append(balanced_accuracy_score(y_true=y_test, y_pred=model.predict(X_test)))
        if verbose:
            print(names[-1], round(BA_scores[-1], 4))

    if verbose:
        print(len(X_test.columns), " features in the dataset")

    return pd.DataFrame(data={"Classifier": names, "BA score": BA_scores})


def standarize(X_train, X_test):
    ss = StandardScaler()
    X_train_sc = pd.DataFrame(ss.fit_transform(X_train), index=X_train.index, columns=X_train.columns)
    X_test_sc = pd.DataFrame(ss.transform(X_test), index=X_test.index, columns=X_test.columns)
    return X_train_sc, X_test_sc


def delete_corr(X, thresh=0.98, verbose=False):
    X = X.copy()
    cor_matrix = X.corr().abs()
    if verbose:
        plt.figure(figsize=(12, 12))
        sns.heatmap(cor_matrix, vmin=0.5, vmax=1)
        plt.show()
    upper_tri = cor_matrix.where(np.triu(np.ones(cor_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper_tri.columns if any(upper_tri[column] >= thresh)]
    X_cleaned = X.drop(columns=to_drop)
    return X_cleaned


def delete_multicollinear(X_train, vif_thresh=10):
    X_train = X_train.copy()
    cols_dropped = []
    while True:
        variables = X_train.columns
        vif = [variance_inflation_factor(X_train[variables].values, X_train.columns.get_loc(var)) for var in X_train.columns]
        max_vif = max(vif)
        if max_vif > vif_thresh:
            maxloc = vif.index(max_vif)
            col_drop = X_train.columns[maxloc]
            # print(f'Dropping {col_drop} with vif={max_vif}')
            X_train = X_train.drop(col_drop, axis=1)
            cols_dropped.append(col_drop)
        else:
            break
        print(X_train.shape[1])
    print(X_train.shape[1], "features left")
    return X_train

def test_univariates(X_train, y_train, X_test, y_test, k=5):
    result = pd.DataFrame(columns=['Classifier', 'BA score', 'Data normalization', 'method', 'k', 'variables'])
    for is_scaled in [False, True]:
        if is_scaled:
            mm = MinMaxScaler()
            X_train = mm.fit_transform(X_train)
            X_test = mm.transform(X_test)
        for method in [chi2, f_classif, mutual_info_classif]:
            kbest = SelectKBest(method, k=k)
            kbest = SelectKBest(method, k=k)
            kbest.fit(X_train, y_train)
            variables = np.arange(0, len(kbest.get_support()))[kbest.get_support()]
            X_train_selected = pd.DataFrame(kbest.transform(X_train), columns=variables)
            X_test_selected = pd.DataFrame(kbest.transform(X_test), columns=variables)
            res = get_models_ba(X_train_selected,
                                X_test_selected,
                                y_train,
                                y_test,
                                verbose=False
                               )
            res['Data normalization'] = is_scaled
            res['method'] = method.__name__
            res['k'] = k
            res['variables'] = json.dumps(variables.tolist())
            result = pd.concat((result, res), ignore_index=True)
            print(f'Scaled {is_scaled}, method {method.__name__}, {len(variables)} variables\n{res}')
    return result

def drop_constants(df, df2):
    to_drop = df.columns[df.var()==0]
    return df.drop(to_drop, axis=1), df2.drop(to_drop, axis=1) 


def get_models_ba_with_hyperparameters_tuning(X_train, X_test, y_train, y_test, logistic_args={}, RF_args={},
                                              AdaBoost_args={}, LGBM_args={}, XDB_args={}, verbose=True):

    X_test = X_test.copy()
    X_test = X_test.loc[:, X_train.columns]
    
    models = (LogisticRegression(),
              RandomForestClassifier(),
              AdaBoostClassifier(),
              LGBMClassifier(),
              XGBClassifier())
    names = []
    valid_accuracy_score = []
    best_params = []
    BA_scores = []
    
    for model, model_param_grid in zip(models, [logistic_args, RF_args, AdaBoost_args, LGBM_args, XDB_args]):
        gscv = GridSearchCV(model, param_grid=model_param_grid, verbose=verbose)
        gscv.fit(X_train, y_train)
        valid_accuracy_score.append(gscv.cv_results_['mean_test_score'].max())
        BA_scores.append(balanced_accuracy_score(y_test, gscv.best_estimator_.predict(X_test)))
        best_params.append(gscv.best_params_)
        names.append(model.__class__.__name__)
        if verbose:
            print(names[-1], 'Valid:', round(valid_accuracy_score[-1], 4), 'Test:', round(BA_scores[-1], 4))
    if verbose:
        print(len(X_test.columns), " features in the dataset")

    return pd.DataFrame(data={"Classifier": names, "best params": best_params,
                              "Valid score": valid_accuracy_score, "BA score": BA_scores})

def estimator_wrappers(X_train, X_test, y_train, y_test, k=50):
    rf = RandomForestClassifier(max_depth=5)
    ab = AdaBoostClassifier()
    lr = LogisticRegression(penalty='l1', solver='liblinear')
    
    results = pd.DataFrame(columns=['Classifier', 'BA score', 'Wrapper'])
    
    for pre_model in [rf, ab, lr]:
        sfm = SelectFromModel(pre_model, max_features=k)
        sfm.fit(X_train, y_train)
        variables = np.arange(0, len(X_train.columns))[sfm.get_support()]
        X_train_sfm = pd.DataFrame(sfm.transform(X_train), columns=variables)
        X_test_sfm = pd.DataFrame(sfm.transform(X_test), columns=variables)
        
        res = get_models_ba(
                X_train_sfm,
                X_test_sfm,
                y_train,
                y_test,
                verbose=False
        )
        res['Wrapper'] = pre_model.__class__.__name__
        results = pd.concat((results, res), ignore_index=True)
    return results