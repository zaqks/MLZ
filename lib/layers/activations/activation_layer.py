from .activation_base import *


# hyperbolic
class TanH(ActivationLayer):
    def __init__(self):
        def tanh(x): return np.tanh(x)
        def tanh_prime(x): return 1-np.tanh(x)**2

        super().__init__(tanh, tanh_prime)


# linear
class Linear(ActivationLayer):
    def __init__(self):
        def linear(x): return x
        def linear_prime(x): return 0

        super().__init__(linear, linear_prime)




class Sigmoid(ActivationLayer):
    def __init__(self):
        def sig(x):
            return  1/(1+np.exp(-x))

        def sig_prime(x):
            s = sig(x)
            return s*(1-s)

        super().__init__(sig, sig_prime)