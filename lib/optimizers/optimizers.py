from .base_optimizer import BaseOptimizer
import numpy as np


class GradientDescent(BaseOptimizer):
    def __init__(self):
        def formula(self, a, d_O, indx): return a*d_O
        super().__init__(formula)


class MomentumOptimizer(BaseOptimizer):
    def __init__(self):
        def formula(self, a, d_O, indx):
            # calc the new v
            self.v[indx] = self.B1 * self.v[indx] + (1-self.B1) * d_O
            #
            return a*self.v[indx]

        super().__init__(formula)
