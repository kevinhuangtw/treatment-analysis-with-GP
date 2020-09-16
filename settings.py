# Samples
N = 1600
SPLIT_RATIO = 1000 / 1600

# Experiments
D = [9, 20]
EXP_NO = [1, 2, 3]
NOISE_SIGMA = [0.1, 0.5]
DEMEAN = [True, False]
SAMPLE_NUM_PER_SETTING = 100
# 12 settings, 10 sample size ~= 1 HR
# 1 sample size: 360 -> 180 -> 90 sec
# [9, 20], [2], [0.1, 0.5], [T, F], 20: 1240 sec ~= 21 min
# [9, 20], [1, 2, 3], [0.1, 0.5], [T, F], 100: 19590 sec ~= 326.5 min ~ 5 hr

# GP
import gpflow


def create_kernel(d):
    lengthscales = {
        9: [0.1] * 9,
        20: [0.1] * 9 + [1.0] * 11
    }
    # Create Kernels
    k1 = gpflow.kernels.SquaredExponential(lengthscales=lengthscales[d])
    # k2 = gpflow.kernels.Linear()
    k = k1
    return k


# Utils
def to1(col):
    """
    :return: 1 if equality is satisfied else 0
    """
    return col.astype(int)
