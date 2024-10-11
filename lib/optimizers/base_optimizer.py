import numpy as np


class BaseOptimizer:
    def __init__(self,  formula):
        self.formula = formula

        self.B1 = 0.9
        self.B2 = 0.999
        self.E = 10e-8

        self.v = None  # w, b, k
        self.s = None  # w, b, k

    def init_vs(self, w, b, k):
        self.v = [np.zeros_like(_) if type(_) != type(None)
                  else None for _ in [w, b, k]]
        self.s = self.v.copy()
