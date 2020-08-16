# %%
import time
import numpy as np
import pandas as pd

pd.options.mode.chained_assignment = None

from sklearn.metrics import confusion_matrix
from data_generation import generate_data
from gp_prediction import gp_predict

# %%
if __name__ == "__main__":
    start_time = time.time()

    record = pd.DataFrame(columns=[
        "N", "D", "EXP_NO", "NOISE_SIGMA", "SAMPLE_NO",
        "ACC", "TN", "FP", "FN", "TP", "R2"])
    record_i = 0

    # %% settings
    n = 1600
    split_ratio = 1000 / 1600
    sample_size = 1  # 10 sample size ~= 1 HR

    test_cases = [(1, 9, 0.1)]  # exp_no, d, noise_sigma

    for exp_no, d, noise_sigma in test_cases:
        for sample_no in range(sample_size):  # sample size
            # %% create predictions
            data = generate_data(n, d, exp_no, noise_sigma)
            prediction = gp_predict(data, n, d,
                                    exp_no, noise_sigma, split_ratio)

            # %% calculate scores
            tn, fp, fn, tp = confusion_matrix(
                prediction["t"], prediction["t_hat"]).ravel()
            acc = (tn + tp) / (tn + fp + fn + tp)

            r = np.corrcoef(prediction["te"],
                            prediction["te_hat"])[0, 1]
            r2 = r ** 2

            # %% add records
            record.loc[record_i] = [n, d, exp_no,
                                    noise_sigma, sample_no,
                                    acc, tn, fp, fn, tp, r2]
            record_i += 1
    else:
        record.to_csv("records.csv")

    print(time.time() - start_time)
