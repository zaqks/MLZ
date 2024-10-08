import numpy as np


class BaseOptimizer:
    def __init__(self,  formula):
        self.formula = formula

        self.B1 = 0.9
        self.B2 = 0.999
        self.E = 10e-8

        # attrs
        self.v = None  # w, b, k
        self.s = None  # w, b, k

    def init_vs(self, w_s, b_s, k_s):

        args = [w_s, b_s, k_s]
        args = [np.zeros(_) if _ else None for _ in args]

        self.v = args
        self.s = args
