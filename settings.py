# Const
N = 1000
D = 20
EXP = "exp1"

# Utils
import time


def file_suffix():
    return "_{}_{}".format(EXP, time.strftime("%m%d_%H%M%S"))


def to1(col):
    """
    :return: 1 if equality is satisfied else 0
    """
    return col.astype(int)
