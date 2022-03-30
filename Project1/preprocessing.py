import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from statsmodels.stats.outliers_influence import variance_inflation_factor
import matplotlib.pyplot as plt
import seaborn as sns

class Preprocessor:
    
    def __init__(self, impute_strategy='most_frequent'):
        self.imputer = SimpleImputer(strategy=impute_strategy)
        self.categorical_columns_values = None
        self.collinear_cols = None
        self.vif_thresh = None
        
    def impute_fit(self, X):
        self.imputer.fit(X)
    
    def impute_transform(self, X):
        return self.imputer.transform(X)
    
    def impute_fit_transform(self, X):
        return self.imputer.fit_transform(X)
    
    #OHE: modified to drop value 'unknown' by default
    
    def one_hot_encoding_fit(self, X, drop_values={}, explicit_columns_values={}):
        """
        drop_values - dict <column_name: value to drop>
        If any column is not in the dict, value 'unknown' is tried to be dropped
        explicit_columns_values - dict <column name: values expected in the column>.
        If any column is not in the dict, the unique values are determined on basis of data
        """
        self.categorical_columns_values = {c: set(X[c].unique())
                                           for c in X.select_dtypes(include='object').columns}
        X_OHE = pd.get_dummies(X)
        for c, c_values in self.categorical_columns_values.items():
            u = drop_values.get(c, 'unknown')
            if u in c_values:
                self.categorical_columns_values[c] = c_values - {u}
            else:
                drop_value = c_values.pop()
                self.categorical_columns_values[c] = c_values
            
    def one_hot_encoding_transform(self, X, sep='_'):
        X_OHE = X.copy()
        if self.categorical_columns_values is None:
            raise ValueError('OHE has not been fit yet')
        for c, c_values in self.categorical_columns_values.items():
            for c_v in c_values:
                X_OHE[f'{c}{sep}{c_v}'] = (X_OHE[c] == c_v).astype(int)
            X_OHE.drop(c, axis=1, inplace=True)
        return X_OHE
    
    def one_hot_encoding_fit_transform(self, X, sep='_'):
        self.one_hot_encoding_fit(X)
        return self.one_hot_encoding_transform(X)
    
    @staticmethod
    def train_test_split(X, y, train_subset_proportion=0.75, keep_y_balance=True):
        if set(X.index) != set(y.index):
            raise AttributeError('Indices in X and y are not indetical')
        n=X.shape[0]
        train_rows_n = int(train_subset_proportion * n)
        test_rows_n = n - train_rows_n 
        if keep_y_balance:
            if ((y.unique()!=0) & (y.unique()!=1)).any():
                raise ValueError('Using keep_y_balance requires y values to be 0 and 1.')
            pos_index = y[y==1].index
            neg_index = y[y==0].index
            train_pos_index = np.random.choice(pos_index, int(train_subset_proportion*len(pos_index)), replace=False)
            train_neg_index = np.random.choice(neg_index, int(train_subset_proportion*len(neg_index)), replace=False)
            test_pos_index = np.array(list(set(pos_index) - set(train_pos_index)))
            test_neg_index = np.array(list(set(neg_index) - set(train_neg_index)))
            train_index = np.concatenate((train_pos_index, train_neg_index))
            test_index = np.concatenate((test_pos_index, test_neg_index))
        else:
            train_index = np.random.choice(y.index, train_rows_n, replace=False)
            test_index = np.array(set(y.index) - set(train_index))
        assert len(set(train_index) & set(test_index)) == 0
        return X.loc[train_index, :], X.loc[test_index, :], y.loc[train_index], y.loc[test_index]
    
    def remove_multicollinearity_fit_transform(self, X, vif_thresh=10):
        #modified to ignore features of type object"
        X_cat=X.select_dtypes(include='object').copy()
        X = X.select_dtypes(exclude='object').copy()
        n = X.shape[1]
        cols_dropped = []
        while True:
            variables = X.columns
            vif = [variance_inflation_factor(X[variables].values, X.columns.get_loc(var)) for var in X.columns]
            max_vif = max(vif)
            if max_vif > vif_thresh:
                maxloc = vif.index(max_vif)
                col_drop = X.columns[maxloc]
                #print(f'Dropping {col_drop} with vif={max_vif}')
                X = X.drop(col_drop, axis=1)
                cols_dropped.append(col_drop)
            else:
                break
        self.collinear_cols = cols_dropped
        self.vif_thresh = vif_thresh
        print(X.shape[1],"numerical features left in dataset ",X_cat.shape[1], " categorical" )
        return X.join(X_cat)
    
    def remove_multicollinearity_transform(self, X):
        if self.vif_thresh is None:
            raise ValueError('Run remove_multicollinearity_fit_transform to fit the module')
        return X.drop(self.collinear_cols, axis=1)
    
    @staticmethod
    def DeleteCorrelated(X,thresh=0.75):
        X=X.copy()
        cor_matrix = X.corr().abs()
        upper_tri = cor_matrix.where(np.triu(np.ones(cor_matrix.shape),k=1).astype(bool))
        to_drop = [column for column in upper_tri.columns if any(upper_tri[column] >= thresh)]
        X_cleaned=X.drop(columns=to_drop)
        return X_cleaned
    
    @staticmethod
    def plot_corr(X):
        plt.figure(figsize=(16,13))
        sns.heatmap(X.corr())
        plt.show()
        
    @staticmethod    
    def class_balancing(X_train,y_train,perc_p=0.2):
        X_train = X_train.copy()
        y_train = y_train.copy()
        
        total_n=len(y_train)
        neg_index=y_train[y_train==0].index
        pos_index=y_train[y_train==1].index
        pos_classes_n = len(pos_index)
        neg_classes_n = total_n-pos_classes_n
        diff_n=pos_classes_n-neg_classes_n

        if (diff_n> perc_p*total_n):
            #if there is above 20 perc point diff, much more of neg_class
            neg_c = np.random.choice(neg_index, size=abs(diff_n), replace=True)
            neg_c = np.hstack([neg_c,neg_index])
            pos_c = pos_index
        elif (diff_n< -perc_p*total_n):
            #if pos_class is sig_bigger
            pos_c = np.random.choice(pos_index, size=abs(diff_n), replace=True)
            pos_c = np.hstack([pos_c,pos_index])
            neg_c = neg_index
        else:
            neg_c = neg_index
            pos_c = pos_index
            
        X_train = X_train.loc[np.hstack([pos_c, neg_c])]
        y_train = y_train.loc[np.hstack([pos_c, neg_c])]
        print("Training dataset has now ", len(y_train), "obervations.",
              (y_train.mean())*100," percent is in positive group.")
            
        return (X_train,y_train)