import numpy as np
from abc import ABC, abstractmethod
from statsmodels.tools import add_constant
from scipy.special import expit
from tqdm.notebook import tqdm

class Optimizer(ABC):
    
    def __init__(self):
        self.name = None
    
    @staticmethod
    def do_early_stop(loss_list, check_no_progress_epochs=10):
        if len(loss_list)<check_no_progress_epochs:
            return False
        last_loss_list = loss_list[-(check_no_progress_epochs+1):]
        for l_i, loss in enumerate(last_loss_list):
            if l_i+1>=check_no_progress_epochs:
                return True
            if loss > last_loss_list[l_i+1]:
                return False
    
    @staticmethod
    def binary_cross_entropy_loss(y_true, y_prob):
        n = y_true.shape[0]
        loss_value = -(np.log(y_prob[y_true]).sum() + np.log(1-y_prob[y_true==0]).sum())/n
        return loss_value
    
    @abstractmethod
    def check_arguments(self, kwargs) -> None:
        pass
    
    @abstractmethod
    def train(self, X, y) -> None:
        pass
    
    @abstractmethod
    def predict(self, X) -> np.ndarray:
        pass
        
class GradientDescent(Optimizer):

    def __init__(self, **kwargs):
        super().__init__()
        self.check_arguments(kwargs)
        self.name = 'Gradient Descent'
        self.w = None
        self.losses = None
        self.batch_size = kwargs['batch_size']
        self.epochs = kwargs['epochs']
        self.learning_rate = kwargs['learning_rate']
        self.is_trained = False

    @staticmethod
    def gradients(X, y_true, y_pred):
        n = len(X)
        dw = X.T@(y_pred-y_true)/n
        return dw
    
    @staticmethod
    def check_arguments(kw):
        required_arguments = {'batch_size', 'epochs', 'learning_rate'}
        lacking_arguments = required_arguments - set(kw)
        if len(lacking_arguments):
            raise ValueError(f'Missing required arguments: {lacking_arguments}')
        left_arguments = set(kw) - required_arguments
        if len(left_arguments):
            raise ValueError(f'Unused arguments: {left_arguments}')

    def train(self, X, y):
        X, y = X.copy(), y.copy()
        X = add_constant(X)
        n, k = X.shape
        w = np.zeros((k, 1))
        y = y.reshape((-1, 1))
        losses = []
        min_loss = np.inf
        for e in tqdm(range(self.epochs)):
            for i in range((n-1)//self.batch_size+1):
                batch_begin = i*self.batch_size
                batch_end = (i+1)*self.batch_size
                x_batch_subset = X[batch_begin:batch_end, :]
                y_batch_subset = y[batch_begin:batch_end]
                y_pred_prob = expit(x_batch_subset @ w)
                w_deriv = self.gradients(x_batch_subset, y_batch_subset, y_pred_prob)
                w -= self.learning_rate * w_deriv
            cur_loss = Optimizer.binary_cross_entropy_loss(y, expit(X @ w))
            losses.append(cur_loss) 
            if cur_loss < min_loss:
                return_w = w
            if Optimizer.do_early_stop(losses):
                print('Early stopping')
                break
        self.w = return_w
        self.losses = losses
        self.is_trained = True
    
    def predict(self, X_new):
        if not self.is_trained:
            raise ValueError('This model has not been trained yet.')
        X = X_new.copy()
        X = add_constant(X)
        return ((X @ self.w) >= 0).astype(int).reshape((-1,))
    
class IRLS(Optimizer):
    
    def __init__(self, **kwargs):
        super().__init__()
        self.check_arguments(kwargs)
        self.name = 'Iterative Reweighted Least Squares'
        self.w = None
        self.losses = None
        self.epochs = kwargs['epochs']
        self.is_trained = False
        #print("tworzę się")
   
    @staticmethod
    def gradients(X, y_true, y_pred):
        n = len(X)
        dw = X.T@(y_pred-y_true)/n
        return dw
    
    @staticmethod
    def hessian(X,y_pred,weights):
        n=len(X)
        diagonal=(y_pred*(1-y_pred)).flatten()
        S=np.diag(diagonal)
        hes=X.T@S@X
        return hes
    
    @staticmethod
    def check_arguments(kw):
        required_arguments = {'epochs'}
        lacking_arguments = required_arguments - set(kw)
        if len(lacking_arguments):
            raise ValueError(f'Missing required arguments: {lacking_arguments}')
        left_arguments = set(kw) - required_arguments
        if len(left_arguments):
            raise ValueError(f'Unused arguments: {left_arguments}')
            
    def train(self, X, y):
        X, y = X.copy(), y.copy()
        X = add_constant(X)
        n, k = X.shape
        w = np.zeros((k, 1))
        y = y.reshape((-1, 1))
        losses = []
        min_loss = np.inf
        for e in tqdm(range(self.epochs)):
            y_pred_prob = expit(X @ w)
            w_deriv = self.gradients(X, y, y_pred_prob)
            w_hess= self.hessian(X,y_pred_prob,w)
            w -= w_hess @ w_deriv
            cur_loss = Optimizer.binary_cross_entropy_loss(y, expit(X @ w))
            losses.append(cur_loss) 
            #print("działam")
            if cur_loss < min_loss:
                return_w = w
            if Optimizer.do_early_stop(losses):
                print('Early stopping')
                break
        self.w = return_w
        self.losses = losses
        self.is_trained = True 
        
    def predict(self, X_new):
        if not self.is_trained:
            raise ValueError('This model has not been trained yet.')
        X = X_new.copy()
        X = add_constant(X)
        return ((X @ self.w) >= 0).astype(int).reshape((-1,))
    
