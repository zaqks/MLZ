from enum import Enum

from .activation import *
from .funcs import *



FUNCS = [relu, sigmoid, linear]


class ActvFuncs(Enum):
    RELU = 0
    SIG = 1
    LINEAR  =2 

    def get(i):
        return FUNCS[i.value]
