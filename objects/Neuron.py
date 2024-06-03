class Neuron:
    def __init__(self, weights, bias):
        self.weights = weights
        self.bias = bias

    def output(self, inpt):
        rslt = 0
        for i in range(inpt.__len__()):     
            rslt += inpt[i] * self.weights[i]

        return rslt + self.bias
    