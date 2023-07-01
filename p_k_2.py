from model import load_data
import numpy as np

gamma = 0.2
C_inv_shear = np.array([[1 + gamma * gamma, -gamma], [-gamma, 1]])

f1 = 3300
f2 = lambda invariant: - 2 / invariant ** 2 

def piola_kirchgoff_2(f1, f2, C_inv, miu=6600, H=1):

    T = miu * H * (f1 * np.eye(2) + f2 * C_inv) 
    return np.array(T)

def save_pk2_anl_shear():
    a, b, X_val, c, d, e = \
            load_data(split=True, extended_data=True)
    
    piola_kirchgoff_2_anl = []
    for invariant in X_val["I2"]:
        piola_kirchgoff_2_anl.append(piola_kirchgoff_2(f1, f2(invariant),
                                                        C_inv=C_inv_shear))
    
    np.save("piola_kirchhoff_analytical_val.npy", piola_kirchgoff_2_anl)

lambda1 = lambda I1, I2: I1 / 2 + ((I1 / 2) ** 2 - I2) ** 0.5
lambda2 = lambda I1, I2: I1 / 2 - ((I1 / 2) ** 2 - I2) ** 0.5

C_inv_ras = lambda lambda1, lambda2: np.array([[lambda1 ** (-2), 0], [0, lambda2 ** (-2)]])

def save_pk2_anl():
    a, b, X_val, c, d, e = \
            load_data(split=True, extended_data=True, validation=True)

    piola_kirchgoff_2_anl = []
    for invariant in X_val.values:

        piola_kirchgoff_2_anl.append(
            piola_kirchgoff_2(f1, f2(invariant[1]),
                                C_inv=(C_inv_ras(lambda1(invariant[0], invariant[1]),
                                lambda2(invariant[0], invariant[1])))))
    
    piola_kirchgoff_2_anl = np.array(piola_kirchgoff_2_anl)
    print(piola_kirchgoff_2_anl[:5])
    print(piola_kirchgoff_2_anl.shape)
    np.save("piola_kirchhoff_analytical_val.npy", piola_kirchgoff_2_anl)

if __name__ == "__main__":

    # _, _, X_val, _, _, _ = \
    #         load_data(split=True,
    #                     extended_data=True,
    #                     validation=True)
    
    # C_arr = []
    # for invariant in X_val.values:
    #     # print(invariant[0], invariant[1])
    #     C_arr.append(C_inv_ras(lambda1(invariant[0], invariant[1]),
    #                         lambda2(invariant[0], invariant[1])))
    
    # np.save("C_inv.npy", C_arr)
   
   save_pk2_anl()
#     for invariant in X_val.values:
#         print(C_inv_ras(lambda1(invariant[0], invariant[1]),
#                         lambda2(invariant[0], invariant[1])))