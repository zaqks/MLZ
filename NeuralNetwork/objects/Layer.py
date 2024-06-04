from .Neuron import Neuron


class Layer:
    def __init__(self, n, id=-1):  # n is the neurons number
        self.id = id

        self.__N = n
        self.__neurons = []

        for i in range(n):
            self.__neurons.append(Neuron([], 0))


    def count_neurons(self):
        return self.__N

    def get_neurons(self):
        return self.__neurons



    