from .Neuron import Neuron, Funcs


class Layer:
    def __init__(self, n, m, activation=None):
        self.neurons = [Neuron(n, activation) for _ in range(m)]

    def get_activated_outputs(self, vals):
        return [neuron.get_activated_output(vals) for neuron in self.neurons]


    def get_neurons(self):
        return self.neurons