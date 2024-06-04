from NeuralNetwork import Network


INPT = [1, 2]

nn = Network(2, [4], 2)
#nn.print()
print(nn.frwrd_prbg(INPT))
