import numpy as np
from scipy.linalg import norm

def analytical_metric(dataset):

    lambda1 = lambda I1, I2: I1 / 2 + ((I1 / 2) ** 2 - I2) ** 0.5
    lambda2 = lambda I1, I2: I1 / 2 - ((I1 / 2) ** 2 - I2) ** 0.5

    def C_inv_ras(pair_invariants):

        I1, I2 = pair_invariants
        return np.array([[lambda1(I1, I2) ** (-2), 0], [0, lambda2(I1, I2) ** (-2)]])

    f1 = 1
    f2 = lambda invariant: - 2 / invariant ** 2

    def piola_kirchgoff_2(f1, f2, C_inv, miu=6600, H=1):
        T = miu * H * (f1 * np.eye(2) + f2 * C_inv)
        return T

