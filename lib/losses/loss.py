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




# binary cross entropy
class Bce(Loss):
    def __init__(self):
        def  bce(y_true, y_pred):
            return -np.mean(
                y_true*np.log(y_pred) +                 
                (1-y_true)*np.log(1-y_pred)
            )

        def  bce_prime(y_true, y_pred):
            return ((1 -y_true) / (1-y_pred) - y_true / y_pred)/np.size(y_true)
            

        super().__init__(bce, bce_prime)

