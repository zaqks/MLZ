from ..optimizers.optimizers import GradientDescent


class Layer:
    def __init__(self, optimizer=None):
        self.input = None
        self.output = None        

        self.optimizer = optimizer

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
        self.optimizer.init_vs(
            self.weights.shape, self.biases.shape, self.kernels.shape)
