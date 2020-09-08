# %%
import numpy as np
from scipy import stats, integrate


# %%
def norm_cdf_diff(x, mean_1, std_1, mean_2, std_2):
    result = (stats.norm.cdf(x, loc=mean_1, scale=std_1)
              - stats.norm.cdf(x, loc=mean_2, scale=std_2))
    return result


def integrate_norm_cdf_diff(x, mean_1, std_1, mean_2, std_2):
    result = integrate.quad(norm_cdf_diff, -np.inf, x,
                            args=(mean_1, std_1, mean_2, std_2))
    return result[0]


def get_xs(mean_1, std_1, mean_2, std_2, num=10):
    min_x = min(mean_1 - 3 * std_1, mean_2 - 3 * std_2)
    max_x = max(mean_1 + 3 * std_1, mean_2 + 3 * std_2)
    xs = np.arange(min_x, max_x, (max_x - min_x) / num)
    return xs


def first_order_dominance(mean_1, std_1, mean_2, std_2):
    for x in get_xs(mean_1, std_1, mean_2, std_2):
        if norm_cdf_diff(x, mean_1, std_1, mean_2, std_2) >= 0:
            return False
    else:
        return True


def second_order_dominance(mean_1, std_1, mean_2, std_2):
    for x in get_xs(mean_1, std_1, mean_2, std_2):
        diff = integrate_norm_cdf_diff(x, mean_1, std_1, mean_2, std_2)
        if diff >= 0:
            return False
    else:
        return True


# %%
if __name__ == "__main__":
    args = [1, 1, 0, 1]  # 平均較好，風險一樣
    print(args)
    print(first_order_dominance(*args))  # True
    print(second_order_dominance(*args))  # True

    args = [0, 1, 0, 1.1]  # 平均一樣，風險較小
    print(args)
    print(first_order_dominance(*args))  # False
    print(second_order_dominance(*args))  # True

    args = [0.5, 2, 0, 1]  # 平均較好，可是風險太大
    print(args)
    print(first_order_dominance(*args))  # False
    print(second_order_dominance(*args))  # False
