from .Layer import Layer


class Network:
    def __init__(self, layers_ns):
        self.layers = []

        previous_layer_ns = layers_ns[0]
        self.layers.append(Layer(previous_layer_ns, previous_layer_ns))
        for layer_ns in self.layers_ns[1:]:
            self.layers.append(Layer(previous_layer_ns, layer_ns))
            previous_layer_ns = layer_ns


    def forward_probg(self, inpt):
        pass