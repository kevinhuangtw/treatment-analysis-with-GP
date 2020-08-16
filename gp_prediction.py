# %%
import numpy as np
import gpflow

from gpflow.utilities import print_summary
from progressbar import progressbar


# %%
def gp_predict(data, N, D, EXP_NO, NOISE_SIGMA, SPLIT_RATIO):
    # %% Const
    X_COLS = ["x" + str(i) for i in range(1, D + 1)]
    Y_COLS = ["y"]

    # %% Split Train/Test
    train_num = int(N * SPLIT_RATIO)
    train = data.iloc[:train_num]
    test = data.iloc[train_num:]

    # %% Train GP Function
    def get_trained_gp(x_df, y_df):
        k = gpflow.kernels.SquaredExponential()
        m = gpflow.models.GPR(data=(x_df.values, y_df.values),
                              kernel=k, mean_function=None)
        opt = gpflow.optimizers.Scipy()
        opt_logs = opt.minimize(m.training_loss, m.trainable_variables,
                                options=dict(maxiter=100))
        return m

    # %% Get Trained GP (for z = 0 and z = 1)
    gp_z0 = get_trained_gp(train.loc[train["z"] == 0, X_COLS],
                           train.loc[train["z"] == 0, Y_COLS])
    gp_z1 = get_trained_gp(train.loc[train["z"] == 1, X_COLS],
                           train.loc[train["z"] == 1, Y_COLS])
    # print_summary(gp_z0)
    # print_summary(gp_z1)

    # %% Predict
    for i, row in progressbar(test.iterrows()):
        # predict y0
        mean, var = gp_z0.predict_f(np.atleast_2d(row[X_COLS].values))
        test.loc[i, "y0_hat"] = mean.numpy().item()

        # predict y1
        mean, var = gp_z1.predict_f(np.atleast_2d(row[X_COLS].values))
        test.loc[i, "y1_hat"] = mean.numpy().item()

    # %% Policy Decision
    test["te"] = test["y1"] - test["y0"]
    test["te_hat"] = test["y1_hat"] - test["y0_hat"]

    test["t"] = (test["te"] < 0).astype(int)
    test["t_hat"] = (test["te_hat"] < 0).astype(int)

    return test
