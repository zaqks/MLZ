from .Neuron import Neuron


class Layer:
    def __init__(self, n, m):
        self.neurons = [Neuron(n) for _ in range(m)]

    def get_activated_outputs(self, vals):
        return [neuron.get_activated_output(vals) for neuron in self.neurons]
