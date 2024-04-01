from math import exp


def May_Yin_psi(I1, I2):
    c0 = 5.95e6  # [kPa]
    c1 = 1.48e-3
    return c0 * (exp(c1 * (I1 - 3) ** 2.) - 1)


def Mooney_Rivlin_psi(I1, I2):
    return 0.0221 * (I1 - 3) + 5 * 10**(-8) * (I2 - 3)


def NeoHookean_psi(I1, I2):
    return 10 * (I1 - 3)
