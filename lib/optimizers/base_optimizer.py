from math import pow


class BaseOptimizer:
    def __init__(self,  formula):
        self.formula = formula

        self.B1 = 0.9
        self.B2 = 0.999
        self.E = 10e-8
