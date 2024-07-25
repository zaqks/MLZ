from NeuralNetwork import *
from random import randrange
from COMBSGEN import COMBSGEN as CombsGen

N = 9
WINS = [
    [0, 1, 2],
    [3, 4, 5],
    [6, 7, 8],

    [0, 3, 6],
    [1, 4, 7],
    [2, 5, 8],

    [0, 4, 8],
    [2, 4, 6],
]


def gen_combs():
    cmbs = CombsGen([0, 1], N)
    cmbs.gen_perfect()
    cmbs.gen_combs()
    cmbs = cmbs.get_combs()

    

    rslt = []

    for cmb in cmbs:
        toAdd = [int(_) for _ in cmb]
        if toAdd.count(1) == 3:
            rslt.append(toAdd)

    return rslt


def is_win(val):
    for cmb in WINS:
        win = True
        for i in cmb:  # indxs
            win = win and val[i]
            if not win:
                break
        if win:
            return True

    return False


ntwrk = Network([N, N, 2], NetworkFuncs.RELU)

inout = InOut(ntwrk)
inout.import_data("data/export.json")


COMBS = gen_combs()
for i in range(50):
    for cmb in COMBS:
        rslt = ntwrk.forward_propg(cmb)
        # print(NetworkFuncs.SOFTMAX(rslt))

        expect = [0, 1]
        if is_win(cmb):
            expect = [1, 0]

        ntwrk.changes_collect(expect)

        print("-------------------")
        print(f"cmb {cmb}")
        print(f"rslt {rslt}")
        print(f"expect {expect}")

    ntwrk.back_prop()

inout.export_data("data/export.json")


"""
cmb = [
    0, 1, 0,
    0, 1, 0,
    1, 0, 0
]
rslt = ntwrk.forward_propg(cmb)

print(cmb)
print(rslt)
print(f"win: {rslt[0] > rslt[1]}")
print(f"diga: {is_win(cmb) != rslt[0] > rslt[1]}")

# inout.export_data("data/export.json")
"""
