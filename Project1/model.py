import numpy as np
import optimizers


class LogReg:
    def __init__(self, optimization, **kwargs):
        self.optimization = optimization
        self.kwargs=kwargs
        self.losses = None
        self.is_trained = False
        self.__is_optimization_known(optimization)
        if optimization == 'Gradient Descent':
            self.__optimizer = optimizers.GradientDescent(**self.kwargs)
        elif optimization == 'Stochastic Gradient Descent':
            self.kwargs['batch_size'] = 1
            self.__optimizer = optimizers.GradientDescent(**self.kwargs)
        elif optimization == 'Iterative Reweighted Least Squares':
            self.__optimizer = optimizers.IRLS(**self.kwargs)
        elif optimization == 'Adaptive Moment Estimation':
            self.__optimizer = optimizers.ADAM(**self.kwargs)
            
    @staticmethod
    def __is_optimization_known(o):
        if o not in ['Gradient Descent', 'Stochastic Gradient Descent',
                     'Iterative Reweighted Least Squares', 'Adaptive Moment Estimation']:
            raise ValueError(f'Unknown optimization {o}')

    def train(self, X, y):
        self.__optimizer.train(X, y)
        
    def predict(self, X):
        return self.__optimizer.predict(X)
    
    def get_optimizer_training_losses(self):
        return self.__optimizer.losses
    def get_optimizer_training_w(self):
        return self.__optimizer.w