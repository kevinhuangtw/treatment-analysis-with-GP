# %%
import numpy as np
import pandas as pd

from experiments.exp1 import *

# %% Const
N = 1000
D = 20

# %% Generate Xs
df = pd.DataFrame()

for i in range(1, D + 1):
    col_name = "x{}".format(i)

    if i % 2 != 0:
        df[col_name] = np.random.normal(0, 1, size=N)
    else:
        df[col_name] = np.random.binomial(1, 0.5, size=N)

# %% Generate base, effect
df["base"] = get_base(df)
df["effect"] = get_effect(df)

# %% Generate y0, y1
df["y0"] = df["base"] - 0.5 * df["effect"]
df["y1"] = df["base"] + 0.5 * df["effect"]

# %% Generate z
df["p(z=1)"] = np.exp(df["y0"]) / (1 + np.exp(df["y0"]))
df["z"] = (np.random.rand(N) <= df["p(z=1)"]).astype(int)

# %% Generate noises
df["y0_noise_sigma"] = get_y0_noise_sigma(df)
df["y1_noise_sigma"] = get_y1_noise_sigma(df)

df["y0_noise"] = np.random.normal(size=N) * df["y0_noise_sigma"]
df["y1_noise"] = np.random.normal(size=N) * df["y1_noise_sigma"]

# %% Get Observable y
df["y0_with_noise"] = df.loc[:, ["y0", "y0_noise"]].sum(axis=1)
df["y1_with_noise"] = df.loc[:, ["y1", "y1_noise"]].sum(axis=1)

df["y"] = pd.concat([df.loc[df["z"] == 0, "y0_with_noise"],
                     df.loc[df["z"] == 1, "y1_with_noise"]])

# %%
df.to_csv("data.csv")
