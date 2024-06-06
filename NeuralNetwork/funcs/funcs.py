from math import exp, log, pow


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


#cost 
def cost(out, expct):
    rslt = 0
    length = out.__len__()
    
    for i in range(length):
        rslt += pow( expct[i] - out[i]  ,   2)
    


    return rslt
