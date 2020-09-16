# %%
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("data.csv")

# %%
df[["y0_noise_sigma", "y1_noise_sigma"]].plot.kde()
plt.show()

# %%
df[["y0_noise", "y1_noise"]].plot.kde()
plt.show()
