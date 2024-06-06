from .Neuron import Neuron, ActvFuncs


class Layer:
    def __init__(self, n,  id=-1, actv=ActvFuncs.RELU):  # n is the neurons number
        self.id = id

        self.__N = n
        self.__neurons = []

        self.__actv = actv

        for i in range(n):
            self.__neurons.append(Neuron([], 0, actv=self.__actv))

    def count_neurons(self):
        return self.__N

    def get_neurons(self):
        return self.__neurons

    def output(self, inpt):
        rslt = []
        
        for nrn in self.__neurons:
            rslt.append(nrn.output(inpt))

        return rslt

