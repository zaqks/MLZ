from NeuralNetwork import Network
from NeuralNetwork.funcs.softmax import softmax

INPT = [1, 4]
nn = Network(2, [4, 4], 2)
# nn = Network(2, [4, 4], 2, src="json/out.json")

# nn.show_ids()
# nn.show_params()


print(nn.frwrd_prbg(INPT))

# nn.export()
