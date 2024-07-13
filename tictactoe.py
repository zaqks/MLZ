from NeuralNetwork import *
from random import randrange

N = 50
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
    def gen_comb():
        win = []

        while win.__len__() < 3:
            val = randrange(0, 9)
            if val not in win:
                win.append(val)

        return [int(i in win) for i in range(9)]

    #

    combs = []

    while combs.__len__() < N:
        val = gen_comb()
        if val not in combs:
            combs.append(val)

    return combs


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


ntwrk = Network([9, 9,  2], NetworkFuncs.RELU)

COMBS = gen_combs()
for cmb in COMBS:
    rslt = ntwrk.forward_probg(cmb)
    rslt = NetworkFuncs.SOFTMAX(rslt)
    print(cmb, rslt)


inout = InOut(ntwrk)
inout.export_data("data/export.json")
