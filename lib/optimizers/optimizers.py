from .base_optimizer import BaseOptimizer


class GradientDescent(BaseOptimizer):
    def __init__(self):
        def formula(a, d_0, indx=None): return a*d_0
        super().__init__(formula)


class MomentumOptimizer(BaseOptimizer):
    def __init__(self):
        def formula(a, d_0, indx):            
            # calculate the new v_w and v_b
            print(indx, self.v[indx].shape, d_0.shape)
            self.v[indx] = self.B1*self.v[indx] + (1-self.B1)*d_0

            return a * self.v[indx]

        super().__init__(formula)
