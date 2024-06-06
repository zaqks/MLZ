from NeuralNetwork import *
from NeuralNetwork.funcs import softmax, cost


# nn = Network(2, [4, 4], 2)
nn = Network(1, [2, 2], 2, actv=ActvFuncs.RELU)

# nn.show_ids()
# nn.show_params()



for val in range(10):
    inpt = [val]
    rslt = nn.frwrd_prbg(inpt)
    rslt = softmax(rslt)
    print(rslt)

    if max(rslt) == rslt[0]:
        print(f"{val} is odd")
    else:
        print(f"{val} is even")

    if val % 2:
        cst = cost(rslt, [1, 0])
    else:
        cst = cost(rslt, [0, 1])

    print(f"cost {cst}")

# nn.export()


"""
a neural network that check if a num is odd of even
"""
