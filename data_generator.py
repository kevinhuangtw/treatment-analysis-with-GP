import pandas as pd
import numpy as np
import unittest


def to1(col):
    """
    :return: 1 if equality is satisfied else 0
    """
    return col.astype(int)


class YGenerator:
    def __init__(self, Xs, noise_sigma):
        self.Xs = Xs
        self.noise_sigma = noise_sigma

    def get_base(self):
        raise NotImplementedError()

    def get_effect(self):
        raise NotImplementedError()

    def get_y_noise(self):
        if self.noise_sigma is None:
            raise NotImplementedError()
        return np.random.normal(scale=self.noise_sigma, size=self.Xs.shape[0])


class YGeneratorExp1(YGenerator):
    def __init__(self, Xs, noise_sigma=0.1):
        if len(Xs.columns) < 9:
            raise AttributeError("Xs should have >= 9 cols")
        super().__init__(Xs, noise_sigma)

    def get_base(self):
        result = (self.Xs["x1"] + self.Xs["x3"] + self.Xs["x5"]
                  + self.Xs["x7"] + self.Xs["x8"] + self.Xs["x9"] - 2)
        return result

    def get_effect(self):
        result = 5 * to1(self.Xs["x1"] > 1) - 5
        return result


class YGeneratorExp2(YGenerator):
    def __init__(self, Xs, noise_sigma=0.2):
        if len(Xs.columns) < 9:
            raise AttributeError("Xs should have >= 9 cols")
        super().__init__(Xs, noise_sigma)

    def get_base(self):
        return self.Xs["x1"] * 0

    def get_effect(self):
        result = \
            (4 * to1(self.Xs["x1"] > 1) * to1(self.Xs["x3"] > 0)
             + 4 * to1(self.Xs["x5"] > 1) * to1(self.Xs["x7"] > 0)
             + 8 * self.Xs["x8"] * self.Xs["x9"])
        return result


class YGeneratorExp3(YGenerator):
    def __init__(self, Xs, noise_sigma=0.5):
        if len(Xs.columns) < 9:
            raise AttributeError("Xs should have >= 9 cols")
        super().__init__(Xs, noise_sigma)

    def get_base(self):
        result = 5 * to1(self.Xs["x1"] > 1) - 5
        return result

    def get_effect(self):
        result = 0.5 * (self.Xs["x1"] ** 2 + self.Xs["x2"]
                        + self.Xs["x3"] ** 2 + self.Xs["x4"]
                        + self.Xs["x5"] ** 2 + self.Xs["x6"]
                        + self.Xs["x7"] ** 2 + self.Xs["x8"]
                        + self.Xs["x9"] ** 2 - 11)
        return result


def get_y_generator(exp_no):
    return {
        1: YGeneratorExp1,
        2: YGeneratorExp2,
        3: YGeneratorExp3
    }.get(exp_no)


class YGeneratorTest(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame([1, 0, 1, 0.3, 0, 1.2, 1, -2, 0],
                               index=["x" + str(i) for i in range(1, 10)]).T

    def test_exp1(self):
        ygen = YGeneratorExp1(self.df)  # sigma=0.1
        self.assertEqual(ygen.get_base().item(), -1)  # 1+1+0+1+-2+0-2=-1
        self.assertEqual(ygen.get_effect().item(), -5)  # 0-5
        self.assertTrue(-0.3 < ygen.get_y_noise().item() < 0.3)  # 99.7%

    def test_exp2(self):
        ygen = YGeneratorExp2(self.df)  # sigma=0.2
        self.assertEqual(ygen.get_base().item(), 0)  # 0
        self.assertEqual(ygen.get_effect().item(), 0)  # 4*0*1+4*0*1+2*-2*0=0
        self.assertTrue(-0.6 < ygen.get_y_noise().item() < 0.6)  # 99.7%

    def test_exp3(self):
        ygen = YGeneratorExp3(self.df)  # sigma=0.5
        self.assertEqual(ygen.get_base().item(), -5)  # 0-5
        # 0.5(1^2+0+1^2+0.3+0^2+1.2+1^2+-2+0^2-11)=0.5*-8.5=-4.25
        self.assertEqual(ygen.get_effect().item(), -4.25)
        self.assertTrue(-1.5 < ygen.get_y_noise().item() < 1.5)  # 99.7%


if __name__ == "__main__":
    unittest.main()
