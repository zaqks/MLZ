from objects.Neuron import Neuron



WEIGHTS = [0.1, 0.2, 0.4 ]
BIAS = 2


print(Neuron(WEIGHTS, BIAS).output([1, 2, 3]))
