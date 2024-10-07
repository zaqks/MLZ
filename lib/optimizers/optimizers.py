from .base_optimizer import BaseOptimizer


class GradientDescent(BaseOptimizer):
    def __init__(self):
        def formula(a, x): return a*x
        super().__init__(formula)

