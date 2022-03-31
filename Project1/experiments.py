import pandas as pd
import measures
from model import LogReg
from preprocessing import Preprocessor 
from sklearn.preprocessing import StandardScaler

def preprocess_data(X_train,y_train,X_test,y_test,remove_collinear,balance_classes,rescaling,preprocessor=Preprocessor(),inplace=False):
    if not inplace:
        X_train=X_train.copy()
        X_train=y_train.copy()
        X_test=X_test.copy()
        y_test=y_test.copy()
    if remove_collinear:
        X_train = preprocessor.remove_multicollinearity_fit_transform(X_train)
        X_test = preprocessor.remove_multicollinearity_transform(X_test)
    if balance_classes:
        X_train,X_train=preprocessor.class_balancing(X_train,X_train)
    if rescaling:
        s = StandardScaler()
        X_train = s.fit_transform(X_train)
        X_test = s.transform(X_test)
    return (X_train,y_train,X_test,y_test)


def test_learning_rates(X_train,y_train,X_test,y_test,l_rates,algorithms,n_epochs=250,batch_size=32,beta_1=0.9,beta_2=0.99):
    results=pd.DataFrame(columns=['learning_rate','method','accuracy','recall','precision','F_measure'])
    for learning_rate in l_rates:
        for alg_short,alg_long in algorithms.items():
            if alg_short=='GD':
                model=LogReg(optimization=alg_long, learning_rate=learning_rate, epochs=n_epochs, batch_size=batch_size)
            elif alg_short=='SGD':
                model=LogReg(optimization=alg_long, learning_rate=learning_rate, epochs=n_epochs)
            else:
                model=LogReg(optimization=alg_long, learning_rate=learning_rate,beta_1=0.9,beta_2=0.99,epsilon=1e-5, epochs=n_epochs)
            model.train(X_train, y_train)
            predictions=model.predict(X_test)
            acc,recall,precision,f_measure=measures.get_measures(predictions, y_test)
            this_row=pd.DataFrame({'learning_rate':learning_rate,
                                'method':alg_short,
                                'accuracy':round(acc,3),
                                'recall':round(recall,3),
                                'precision':round(precision,3),
                                'F_measure':round(f_measure,3)},index=[0])
            
            results=pd.concat([results,this_row],ignore_index=True)
    return results

def test_betas(X_train,y_train,X_test,y_test,tested_betas1,tested_betas2,learning_rates,n_epochs=250):
    results=pd.DataFrame(columns=['beta1','beta2','accuracy','recall','precision','F_measure'])
    for lr in learning_rates:
        for beta1 in tested_betas1:
            for beta2 in tested_betas2:
                model=LogReg(optimization='Adaptive Moment Estimation',
                            learning_rate=lr,beta_1=beta1,beta_2=beta2,epsilon=1e-5, epochs=n_epochs)
                model.train(X_train, y_train)
                predictions=model.predict(X_test)
                acc,recall,precision,f_measure=measures.get_measures(predictions, y_test)
                this_row=pd.DataFrame({'lr': lr,
                                       'beta1':beta1,
                                       'beta2':beta2,
                                       'accuracy':acc,
                                       'recall':recall,
                                       'precision':precision,
                                       'F_measure':f_measure},
                                        index=[0])
                results=pd.concat([results,this_row],ignore_index=True)
    return results
def final_comparisson(X_train,y_train,X_test,y_test,models):
    results=pd.DataFrame(columns=['model','accuracy','recall','precision','f_measure'])
    for name,model in models.items():
        if name in ('GD','SGD','IRLS','ADAM'):
            model.train(X_train, y_train)
        else:
            model.fit(X_train,y_train)
        predictions=model.predict(X_test)
        acc,recall,precision,f_measure=measures.get_measures(predictions, y_test)
        this_row=pd.DataFrame({'model':name
                            ,'accuracy':acc
                            ,'recall':recall
                            ,'precision':precision
                            ,'f_measure':f_measure},
                                index=[0])
        results=pd.concat([results,this_row],ignore_index=True)
    return results

