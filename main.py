from NeuralNetwork import Network


ntwrk = Network([2,  1])

rslt = ntwrk.forward_probg([2, 3])
print(rslt)