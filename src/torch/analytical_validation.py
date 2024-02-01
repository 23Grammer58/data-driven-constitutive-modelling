import numpy as np
from scipy.linalg import norm


lambda1 = lambda I1, I2: I1 / 2 + ((I1 / 2) ** 2 - I2) ** 0.5
lambda2 = lambda I1, I2: I1 / 2 - ((I1 / 2) ** 2 - I2) ** 0.5

C_inv_ras = lambda I1, I2: np.array([[lambda1(I1, I2) ** (-2), 0], [0, lambda2(I1, I2) ** (-2)]])

f1 = 1
f2 = lambda invariant: - 2 / invariant ** 2

def piola_kirchgoff_2(f1, f2, C_inv, miu=6600, H=1):
    T = miu * H * (f1 * np.eye(2) + f2 * C_inv)
    return T


def metric(T_arr, T_pred_arr):
    mean = []
    max = []
    for i in range(len(T_arr)):

        if np.any(np.isnan(T_pred_arr[i])) == True or np.any(np.isnan(T_arr[i])) == True:
            continue
        znam = 4 * norm(T_arr[i], ord="fro")
        mean.append((T_pred_arr[i][0, 0] - T_arr[i][0, 0] + T_pred_arr[i][1, 1] - T_arr[i][1, 1]) / znam)
        max.append(np.max(norm(T_pred_arr[i] - T_arr[i])) / znam)

    return np.array(mean), np.array(max)


def calculate_C(invariants):

    lambda1 = lambda I1, I2: I1 / 2 + ((I1 / 2) ** 2 - I2) ** 0.5
    lambda2 = lambda I1, I2: I1 / 2 - ((I1 / 2) ** 2 - I2) ** 0.5

    def C_inv_ras(pair_invariants):

        I1, I2 = pair_invariants
        return np.array([[lambda1(I1, I2) ** (-2), 0], [0, lambda2(I1, I2) ** (-2)]])

    C_arr = [C_inv_ras(invariant_1_2) for invariant_1_2 in invariants]

    return C_arr


if __name__ == "__main__":

    pk2_pred = []
    for f, C in zip(f2_pred, C_inv_arr):
        pk2_pred.append(piola_kirchgoff_2(f1, f, C))

    pk2_pred = np.array(pk2_pred)
    pk2_anl = np.load("piola_kirchhoff_analytical_val.npy")


