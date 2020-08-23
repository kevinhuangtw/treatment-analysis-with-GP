# %%
import pandas as pd
import matplotlib.pyplot as plt

# %%
df = pd.read_csv("records_0816_164621.csv")

#%% ACC
for key, grp in df.groupby(['EXP_NO']):
    grp.boxplot(column="ACC", by=["DEMEAN"])
    plt.show()

#%% R2
for key, grp in df.groupby(['EXP_NO']):
    grp.boxplot(column="R2", by=["DEMEAN"])
    plt.show()
