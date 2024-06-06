from math import exp, log


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


# categorical cross entropy
def cce(inpt, label):
    return -log(inpt[label])


loss = cce([0.7, 0.1 , 0.2], 0)
print(loss)
print("bonjour")
