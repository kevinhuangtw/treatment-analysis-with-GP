# %%
import os
import pandas as pd
import matplotlib.pyplot as plt

# %%
fname = [f for f in os.listdir()
         if f.startswith("data") and f.endswith(".csv")][-1]

df = pd.read_csv(fname)

# %%
df[["y0_noise_sigma", "y1_noise_sigma"]].plot.kde()
plt.show()

# %%
df[["y0_noise", "y1_noise"]].plot.kde()
plt.show()
