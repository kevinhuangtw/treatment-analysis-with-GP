import numpy as np

from settings import to1


def get_base(df):
    return df["x1"] * 0


def get_effect(df):
    result = (4 * to1(df["x1"] > 1) * to1(df["x3"] > 0)
              + 4 * to1(df["x5"] > 1) * to1(df["x7"] > 0)
              + 8 * df["x8"] * df["x9"])
    return result


def get_y0_noise_sigma(df, const=0.1):
    return const + df["z"] + np.sin(df["base"])


def get_y1_noise_sigma(df, const=0.2):
    return const + 2 * df["z"] + np.sin(df["base"] + df["effect"])
