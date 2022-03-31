import numpy as np

def __is_classification_not_binary(y):
    return ((y != 0) & (y != 1)).any()

def __check_initial_conditions(y_test, y_pred):
    if y_test.shape != y_pred.shape:
        raise ValueError(f'Shapes of arguments are not the same. '
                            f'y_test: {y_test.shape}, y_pred: {y_pred.shape}')
    if __is_classification_not_binary(y_pred):
        raise ValueError('y_pred contains non 0/1 values.')
    if __is_classification_not_binary(y_test):
        raise ValueError('y_test contains non 0/1 values.')

def accuracy(y_pred, y_test, check_cond=True):
    if check_cond:
        __check_initial_conditions(y_test, y_pred)
    return (y_pred == y_test).mean()

def recall(y_pred, y_test, check_cond=True):
    """
    Recall = TPR = TP/(TP+FN)
    """
    if check_cond:
        __check_initial_conditions(y_test, y_pred)
    if not y_test.sum():
        raise ZeroDivisionError(f'Recall can not be calculated as there are no positive observations.')
    tp = ((y_pred==1) & (y_pred==y_test)).sum()
    pv = y_test.sum()
    return tp/pv

def precision(y_pred, y_test, check_cond=True):
    """
    Precision = (TP)/(TP+FP)
    """
    if check_cond:
        __check_initial_conditions(y_test, y_pred)
    if not y_pred.sum():
        return 0
#         raise ZeroDivisionError(f'Precision can not be calculated as there are no positive predictions.')
# Error would be relevant here but sometimes we need to have a value of metric even in a non-sense case 
    
    tp = ((y_pred==1) & (y_pred==y_test)).sum()
    pp = y_pred.sum()
    return tp/pp

def f_measure(y_pred, y_test):
    r = recall(y_pred, y_test)
    p = precision(y_pred, y_test)
    return 2*r*p/(r+p)

def get_measures(y_pred, y_test):
    acc=accuracy(y_pred, y_test)
    try:
        rec=recall(y_pred, y_test)
    except ZeroDivisionError:
        rec=np.NAN;
    try:
        prec=precision(y_pred, y_test)
    except ZeroDivisionError:
        prec=np.NAN;
    try:
        f_me=f_measure(y_pred, y_test)
    except ZeroDivisionError:
        f_me=np.NAN;
    return (acc,rec,prec,f_me)