# %%
import os
import pandas as pd
import matplotlib.pyplot as plt

# %%
csv = [f for f in os.listdir() if f.startswith("record")][-1]  # latest
df = pd.read_csv(csv)

#%%
group_cols = ["EXP_NO"]
x_axis_cols = ["D", "NOISE_SIGMA", "DEMEAN"]

#%% ACC
for key, grp in df.groupby(group_cols):
    grp.boxplot(column="ACC", by=x_axis_cols, rot=60)
    plt.show()

#%% R2
for key, grp in df.groupby(group_cols):
    grp.boxplot(column="R2", by=x_axis_cols, rot=60)
    plt.show()
