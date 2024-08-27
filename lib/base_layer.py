class Layer:
    def __init__(self):
        self.input = None
        self.output = None

    def forward(self):
        pass
    
    """
    pass an optimizer in the backward
    """
    def backward(self, E_Y, a):
        # E_Y is the output gradient
        # a is the learning rate
        pass
