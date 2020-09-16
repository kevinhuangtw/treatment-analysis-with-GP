import numpy as np

from settings import to1


def get_base(df):
    result = (df["x1"] + df["x3"] + df["x5"]
              + df["x7"] + df["x8"] + df["x9"] - 2)
    return result


def get_effect(df):
    return 5 * to1(df["x1"] > 1) - 5


def get_y0_noise_sigma(df, const=0.1):
    return const + df["z"] + np.sin(df["base"])


def get_y1_noise_sigma(df, const=0.2):
    return const + 2 * df["z"] + np.sin(df["base"] + df["effect"])
