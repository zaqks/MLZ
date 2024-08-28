import numpy as np


class Loss:
    def __init__(self, func, func_prime):
        self.func = func
        self.prime = func_prime


# mean squared error
class Mse(Loss):
    def __init__(self):
        def mse(y_true, y_pred):
            return np.mean(np.power(y_true-y_pred, 2))


        def mse_prime(y_true, y_pred):
            return 2 * (y_pred - y_true) / np.size(y_true)

        super().__init__(mse, mse_prime)