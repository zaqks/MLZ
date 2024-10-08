from ..optimizers.optimizers import GradientDescent


class Layer:
    def __init__(self, optimizer=None):
        self.input = None
        self.output = None

        self.optimizer = optimizer

        self.weights = None
        self.biases = None
        self.kernels = None

    def forward(self):
        pass

    """
    pass an optimizer in the backward
    """

    def backward(self, E_Y, a):
        # E_Y is the output gradient
        # a is the learning rate
        pass

    def init_optimizer(self):
        shps = [
            self.weights, self.biases, self.kernels
        ]

        shps = [_.shape if _ else None for _ in shps]

        self.optimizer.init_vs(
            *shps)
