from NeuralNetwork import Network, InOut, NetworkFuncs
from COMBSGEN import COMBSGEN


ntwrk = Network([2,  2], activation=NetworkFuncs.LINEAR)
io = InOut(ntwrk).import_data("data/data.json")


cmb = COMBSGEN([0, 1], 2)
cmb.gen_perfect()
cmb.gen_combs()
cmb = cmb.get_combs()

CMBS = [[int(ltr) for ltr in wrd] for wrd in cmb]


for cmb in CMBS[:2]:
    cmb = [0, 1]
    rslt = ntwrk.forward_propg(cmb)

    print("-----------------")
    print(cmb)
    print(rslt)    

    expect = [int(cmb[-1] == 1), int(cmb[-1] == 0)]
    print(f"expect: {expect} [odd, even]")
    print("-----------------")


    break
