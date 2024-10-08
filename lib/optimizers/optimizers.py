from .base_optimizer import BaseOptimizer
import numpy as np


class GradientDescent(BaseOptimizer):
    def __init__(self):
        def formula_w(self, a, d_W):
            return a*d_W

        def formula_b(self, a, d_B):
            return a*d_B


class MomentumOptimizer(BaseOptimizer):
    def __init__(self):
        self.v_w = None
        self.v_b = None

    def init_vw(self, w_s):
        self.v_w = np.zeros(w_s)

    def init_vb(self, b_s):
        self.v_b = np.zeros(b_s)

    def calc_vw(self, d_W):
        self.v_w = self.B1*self.v_w + (1-self.B1)*d_W
        return self.v_w

    def calc_vb(self, d_B):
        self.v_b = self.B1*self.v_b + (1-self.B1)*d_B
        return self.v_b

    def formula_w(self, a, d_W):
        return a*self.calc_vw(d_W)

    def formula_b(self, a, d_B):
        return a*self.calc_vb(d_B)
