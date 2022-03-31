import numpy as np
from abc import ABC, abstractmethod
from statsmodels.tools import add_constant
from scipy.special import expit
from sklearn.utils import shuffle
#from tqdm.notebook import tqdm
from scipy.sparse import diags
from sklearn.metrics import log_loss

import matplotlib.pyplot as plt

class Optimizer(ABC):

    def __init__(self):
        self.name = None
        self.is_trained = False

    @staticmethod
    def do_early_stop(loss_list, check_no_progress_epochs=10):
        if len(loss_list) < check_no_progress_epochs:
            return False
        last_loss_list = loss_list[-(check_no_progress_epochs + 1):]
        for l_i, loss in enumerate(last_loss_list):
            if l_i + 1 >= check_no_progress_epochs:
                return True
            if loss > last_loss_list[l_i + 1]:
                return False

    @staticmethod
    def binary_cross_entropy_loss(y_true, y_prob):
        n = y_true.shape[0]
        loss_value = log_loss(y_true, y_prob)#-(np.log(y_prob[y_true == 1]).sum() + np.log(1 - y_prob[y_true == 0]).sum()) / n
        return loss_value

    @abstractmethod
    def train(self, X, y) -> None:
        pass

    @abstractmethod
    def predict(self, X) -> np.ndarray:
        pass

    @staticmethod
    def gradients(X, y_true, y_pred):
        n = len(X)
        dw = X.T @ (y_pred - y_true) / n
        return dw

    def check_arguments(self, kw, required_arguments) -> None:
        lacking_arguments = required_arguments - set(kw)
        if len(lacking_arguments):
            raise ValueError(f'Missing required arguments: {lacking_arguments}')
        left_arguments = set(kw) - required_arguments
        if len(left_arguments):
            raise ValueError(f'Unused arguments: {left_arguments}')

    def predict(self, X_new):
        if not self.is_trained:
            raise ValueError('This model has not been trained yet.')
        X = X_new.copy()
        X = np.c_[np.ones((X.shape[0], 1)), X]
        return ((X @ self.w) >= 0).astype(int).reshape((-1,))


class GradientDescent(Optimizer):

    def __init__(self, **kwargs):
        super().__init__()
        self.check_arguments(kwargs, {'batch_size', 'epochs', 'learning_rate'})
        self.name = 'Gradient Descent'
        self.w = None
        self.losses = None
        self.batch_size = kwargs['batch_size']
        self.epochs = kwargs['epochs']
        self.learning_rate = kwargs['learning_rate']
        self.is_trained = False

    def train(self, X, y):
        
        X, y = X.copy(), y.copy()
        X = add_constant(X)
        n, k = X.shape
        #pozwala na wrzucanie na ile batchy ma dzielić dataset, bo 32 w jednym datasecie to dużo a w innym mało
        if (self.batch_size<1):
            self.batch_size=int(self.batch_size*n)
        w = np.zeros((k, 1)) # np.random.normal(size=k).reshape((-1, 1))
        return_w = w.copy()
        y = y.reshape((-1, 1))
        losses = []
        min_loss = np.inf
        for e in range(self.epochs):
            X, y = shuffle(X, y, random_state=53)
            for i in range((n - 1) // self.batch_size + 1):
                batch_begin = i * self.batch_size
                batch_end = (i + 1) * self.batch_size
                x_batch_subset = X[batch_begin:batch_end, :]
                y_batch_subset = y[batch_begin:batch_end]
                y_pred_prob = expit(x_batch_subset @ w)
                w_deriv = self.gradients(x_batch_subset, y_batch_subset, y_pred_prob)
                w_diff = -self.learning_rate * w_deriv
                w = w + w_diff
            cur_loss = self.binary_cross_entropy_loss(y, expit(X @ w))
            losses.append(cur_loss)
            if cur_loss < min_loss:
                return_w = w.copy()
                min_loss = cur_loss
            if self.do_early_stop(losses):
                #print('Early stopping')
                break
        self.w = return_w
        self.losses = losses
        self.is_trained = True


class IRLS(Optimizer):

    def __init__(self, **kwargs):
        super().__init__()
        self.check_arguments(kwargs, {'epochs'})
        self.name = 'Iterative Reweighted Least Squares'
        self.w = None
        self.losses = None
        self.epochs = kwargs['epochs']
        self.is_trained = False

    @staticmethod
    def hessian_inv(X, y_pred, weights):
        n = len(X)
        diagonal = (y_pred * (1 - y_pred))
        try:
            X_t = np.multiply(X, diagonal)
            hes_inv = np.linalg.inv(X_t.T @ X)
        except np.linalg.LinAlgError as e:
            print(e)
            hes_inv=1e-3*np.eye(N=X.shape[1])
        return hes_inv

    def train(self, X, y):
        X, y = X.copy(), y.copy()
        X = add_constant(X)
        n, k = X.shape
        w = np.zeros((k, 1))
        return_w = w.copy()
        y = y.reshape((-1, 1))
        losses = []
        min_loss = np.inf
        for e in range(self.epochs):
            y_pred_prob = expit(X @ w)
            w_deriv = self.gradients(X, y, y_pred_prob)
            w_hess_inv = self.hessian_inv(X, y_pred_prob, w)
            w -= w_hess_inv @ w_deriv
            cur_loss = self.binary_cross_entropy_loss(y, expit(X @ w))
            losses.append(cur_loss)
            if cur_loss < min_loss:
                return_w = w.copy()
                min_loss = cur_loss
            if self.do_early_stop(losses):
                #print('Early stopping')
                break
        self.w = return_w
        self.losses = losses
        self.is_trained = True


class ADAM(Optimizer):

    def __init__(self, **kwargs):
        super().__init__()
        self.check_arguments(kwargs, {'epochs', 'learning_rate', 'beta_1', 'beta_2', 'epsilon'})
        self.name = 'Adaptive Moment Estimation'
        self.w = None
        self.losses = None
        self.epochs = kwargs['epochs']
        self.learning_rate = kwargs['learning_rate']
        self.beta_1 = kwargs['beta_1']
        self.beta_2 = kwargs['beta_2']
        self.epsilon = kwargs['epsilon']
        self.is_trained = False

    def train(self, X, y):
        X, y = X.copy(), y.copy()
        X = add_constant(X)
        n, k = X.shape

        w = np.zeros((k, 1))
        y = y.reshape((-1, 1))
        mean = np.zeros((k, 1))
        var = np.zeros((k, 1))

        b1 = self.beta_1
        b2 = self.beta_2
        lr = self.learning_rate

        losses = []
        min_loss = np.inf
        return_w = w.copy()
        for e in range(self.epochs):
            if e > 0:
                 lr *= np.sqrt(e) / np.sqrt(e + 1)
            y_pred_prob = expit(X @ w)
            deriv = self.gradients(X, y, y_pred_prob)
            mean = b1 * mean + (1 - b1) * deriv
            var = b2 * var + (1 - b2) * deriv ** 2
            mean_bias = mean / (1 - b1 ** (e + 1))
            var_bias = var / (1 - b2 ** (e + 1))
            w -= lr * mean_bias / (np.sqrt(var_bias) + self.epsilon)
            cur_loss = self.binary_cross_entropy_loss(y, expit(X @ w))
            losses.append(cur_loss)
            if cur_loss < min_loss:
                return_w = w.copy()
                min_loss = cur_loss
            if self.do_early_stop(losses):
                #print('Early stopping')
                break
        self.w = return_w
        self.losses = losses
        self.is_trained = True
