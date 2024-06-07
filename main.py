from NeuralNetwork import *
from NeuralNetwork.funcs import softmax, cost

# nn = Network(2, [4, 4], 2)
nn = Network(1, [4], 2, actv=ActvFuncs.RELU)

# nn.show_ids()
# nn.show_params()


for val in range(4):
    rslt = nn.frwrd_prbg([val])
    rslt = softmax(rslt)

    if val % 2:
        cst = cost(rslt, [1, 0])
    else:
        cst = cost(rslt, [0, 1])

    # wssh rah y9ol l net
    if rslt[0] > rslt[1]:
        wrd = "odd"
    else:
        wrd = "even"

    print(f"{val} is {wrd}")
    print(f"{val} {rslt} {cst}\n")


nn.export()


"""
a neural network that check if a num is odd or even
"""
