from math import exp

def softmax(inpt):
    rslt = []

    exps = []
    sum = 0

    for i in inpt:
        i = exp(i)
        exps.append(i)

        sum += i

    for i in exps:
        rslt.append(i / sum)

    return rslt

VALS = [1.3, 5.1,2.2, 0.7, 1.1 ]
print(softmax(VALS))
