from .base_optimizer import BaseOptimizer
import numpy as np


class GradientDescent(BaseOptimizer):
    def __init__(self):
        def formula(a, d_O, indx): return a*d_O
        super().__init__(formula)


class MomentumOptimizer(BaseOptimizer):
    def __init__(self):
        def formula(a, d_O, indx):
            # calc the new v
            self.v[indx] = self.B1 * self.v[indx] + (1-self.B1) * d_O
            #
            return a*self.v[indx]

        super().__init__(formula)

# Root Mean Squared Propagation


class RMSPropOptimizer(BaseOptimizer):
    def __init__(self):
        def formula(a, d_O, indx):
            self.s[indx] = self.B2 * self.s[indx] + \
                (1-self.B2) * np.power(d_O, 2)
            #
            return a*d_O/np.sqrt(self.s[indx]+self.E)

        super().__init__(formula)


class AdamOptimizer(BaseOptimizer):
    def __init__(self):
        def formula(a, d_O, indx):

            self.v[indx] = self.B1 * self.v[indx] + (1-self.B1) * d_O
            self.s[indx] = self.B2 * self.s[indx] + \
                (1-self.B2) * np.power(d_O, 2)            

            return a*self.v[indx]/np.sqrt(self.s[indx]+self.E)

        super().__init__(formula)
