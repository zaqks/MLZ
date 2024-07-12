from NeuralNetwork import Network


ntwrk = Network([6, 4, 2])

rslt = ntwrk.forward_probg([1, 2, 3, 4, 5, 6])
print(rslt)