from math import pow


class BaseOptimizer:
    def __init__(self,  formula):
        self.formula = formula

        self.B1 = 0.9
        self.B2 = 0.999
        self.E = 10e-8

        # attrs
        self.v = None  # w, b, k
        self.s = None  # w, b, k

    def init_grads(self, w, b, k):
        if None in [self.v, self.s]:            
            args = [w, b, k]
            
            self.v = args
            self.s = args
