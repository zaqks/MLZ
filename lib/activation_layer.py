from .activation_base import *


# hyperbolic
class TanH(ActivationLayer):
    def __init__(self):
        def tanh(x): return np.tanh(x)
        def tanh_prime(x): return 1-np.tanh(x)**2

        super().__init__(tanh, tanh_prime)
