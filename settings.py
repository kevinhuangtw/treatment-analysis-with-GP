# Samples
N = 1600
SPLIT_RATIO = 1000 / 1600

# Experiments
D = [9, 20]
EXP_NO = [2]
NOISE_SIGMA = [0.1]
DEMEAN = [True, False]
SAMPLE_NUM_PER_SETTING = 5
# 12 settings, 10 sample size ~= 1 HR
# 1 sample size: 360 -> 180 -> 90 sec

# GP
import gpflow

def create_kernel(d):
    lengthscales = {
        9: [0.1] * 9,
        20: [0.1] * 9 + [1.0] * 10
    }
    # Create Kernels
    k1 = gpflow.kernels.SquaredExponential(lengthscales=lengthscales[d])
    # k2 = gpflow.kernels.Linear()
    k = k1
    return k
