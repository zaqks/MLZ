from NeuralNetwork import *
from NeuralNetwork.funcs import softmax, cost

INPT = [1, 4]
# nn = Network(2, [4, 4], 2)
nn = Network(2, [4, 4], 2, src="json/out.json", actv=ActvFuncs.RELU)

# nn.show_ids()
# nn.show_params()


rslt = nn.frwrd_prbg(INPT)
print(rslt)
print(softmax(rslt))

# nn.export()
