# %%
import numpy as np
import pandas as pd

from data_generator import get_y_generator

# %%
if __name__ == "__main__":
    # %% Const
    N = 1000
    D = 20
    EXP_NO = 1
    NOISE_SIGMA = 0.5


# %%
def generate_data(N, D, EXP_NO, NOISE_SIGMA):
    # %% Generate Xs
    df = pd.DataFrame()

    for i in range(1, D + 1):
        col_name = "x{}".format(i)

        if i % 2 != 0:
            df[col_name] = np.random.normal(0, 1, size=N)
        else:
            df[col_name] = np.random.binomial(1, 0.5, size=N)

    # %% Generate Base, Effect
    gen = get_y_generator(EXP_NO)(Xs=df, noise_sigma=NOISE_SIGMA)
    df["base"] = gen.get_base()
    df["effect"] = gen.get_effect()
    df["y_noise"] = gen.get_y_noise()

    # %% Generate y0, y1
    df["y0"] = df["base"] - 0.5 * df["effect"]
    df["y1"] = df["base"] + 0.5 * df["effect"]

    # %% Generate p(z=1), z
    df["p(z=1)"] = np.exp(df["y0"]) / (1 + np.exp(df["y0"]))
    p_rand = np.random.rand(N)
    df["z"] = (p_rand <= df["p(z=1)"]).astype(int)

    # %% Get Observable y (with noise)
    df["y"] = pd.concat([df["y0"].loc[df["z"] == 0],
                         df["y1"].loc[df["z"] == 1]])
    df["y"] = df["y"] + df["y_noise"]

    # %%
    return df
