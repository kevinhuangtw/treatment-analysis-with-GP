# %%
import time
import settings
import numpy as np
import pandas as pd

pd.options.mode.chained_assignment = None

from sklearn.metrics import confusion_matrix
from data_generation import generate_data
from gp_prediction import gp_predict

import itertools
import multiprocessing as mp
import tensorflow as tf

tf.config.experimental.set_memory_growth(
    tf.config.experimental.get_visible_devices("GPU")[0], True)


# %%
def run(d, exp_no, noise_sigma, demean=True, sample_no=0):
    n = settings.N
    split_ratio = settings.SPLIT_RATIO

    try:
        # %%
        data = generate_data(n, d, exp_no, noise_sigma)
        prediction = gp_predict(data, split_ratio, demean)

        tn, fp, fn, tp = confusion_matrix(
            prediction["t"], prediction["t_hat"]).ravel()
        acc = (tn + tp) / (tn + fp + fn + tp)

        # %%
        r = np.corrcoef(prediction["te"],
                        prediction["te_hat"])[0, 1]
        r2 = r ** 2

        result = [n, d, exp_no, noise_sigma, demean, sample_no,
                  acc, tn, fp, fn, tp, r2]
    except Exception as e:
        result = [n, d, exp_no, noise_sigma, demean, sample_no] + [np.nan] * 6

    finally:
        return result


# %%
if __name__ == "__main__":
    start_time = time.time()

    # %% settings
    args = list(itertools.product(
        settings.D, settings.EXP_NO, settings.NOISE_SIGMA, settings.DEMEAN,
        list(range(settings.SAMPLE_NUM_PER_SETTING))))

    # %%
    with mp.Pool(3) as p:
        result = p.starmap(run, args)

    record = pd.DataFrame(result, columns=[
        "N", "D", "EXP_NO", "NOISE_SIGMA", "DEMEAN", "SAMPLE_NO",
        "ACC", "TN", "FP", "FN", "TP", "R2"])

    record.to_csv("records_{}_{}.csv".format(
        time.strftime("%m%d_%H%M%S"),
        "".join([str(no) for no in settings.EXP_NO])))

    print(time.time() - start_time)
