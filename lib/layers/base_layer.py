from ..optimizers.optimizers import GradientDescent


class Layer:
    def __init__(self, optimizer=None, w=None, b=None, k=None):
        self.input = None
        self.output = None

        self.optimizer = optimizer

        self.weights = w
        self.biases = b
        self.kernels = k

    def forward(self):
        pass

    """
    pass an optimizer in the backward
    """

    def backward(self, E_Y, a):
        # E_Y is the output gradient
        # a is the learning rate
        pass
