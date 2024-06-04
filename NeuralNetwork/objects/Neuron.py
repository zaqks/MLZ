class Neuron:
    def __init__(self, weights, bias, id=-1):
        self.id = id
        
        self.__weights = weights
        self.__bias = bias

    def output(self, inpt):
        rslt = 0
        for i in range(inpt.__len__()):
            rslt += inpt[i] * self.__weights[i]

        return rslt + self.__bias


    