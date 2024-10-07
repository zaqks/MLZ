from .base_optimizer import BaseOptimizer


class GradientDescent(BaseOptimizer):
    def __init__(self):
        def formula(a, g): return a*g
        super().__init__(formula)


class MomentumOptimizer(BaseOptimizer):
    def __init__(self):
        def formula(a, g):
            return

        super().__init__(formula)
