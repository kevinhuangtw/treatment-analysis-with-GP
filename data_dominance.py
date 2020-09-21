# %%
import os
import settings
import pandas as pd

from progressbar import progressbar
from dominance import first_order_dominance, second_order_dominance


# %%
def generate_data_dominance(fname=None):
    # %%
    if fname is None:
        fname = [f for f in os.listdir()
                 if f.startswith("data")][-1]  # latest

    df = pd.read_csv(fname)
    data_dominance = []

    for i, row in progressbar(df.iterrows()):
        args = [row["y1"], row["y1_noise_sigma"],
                row["y0"], row["y0_noise_sigma"]]

        MD = int(row["y1"] >= row["y0"])
        SSD = int(second_order_dominance(*args))
        FSD = int(first_order_dominance(*args))
        data_dominance.append(args + [MD, SSD, FSD])

    # %%
    df_dominance = pd.DataFrame(data_dominance, columns=[
        "y1", "y1_noise_sigma", "y0", "y0_noise_sigma", "MD", "SSD", "FSD"])
    fname = "dominance_data" + settings.file_suffix() + ".csv"
    df_dominance.to_csv(fname)
    return fname
