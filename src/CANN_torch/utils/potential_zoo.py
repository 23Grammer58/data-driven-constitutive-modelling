from math import exp
import torch


def May_Yin_psi(I1, I2):
    c0 = 5.95e6  # [kPa]
    c1 = 1.48e-3
    return c0 * (exp(c1 * (I1 - 3) ** 2.) - 1)


def Mooney_Rivlin_psi(I1, I2):
    return 0.0221 * (I1 - 3) + 5 * 10**(-8) * (I2 - 3)


def NeoHookean_psi(I1, I2):
    return 10 * (I1 - 3)

def get_psi(w, terms:int = 6, activation_func = ("exp", "log"), p:int = 3):
    psi = "add view of potential in potential_zoo"
    if terms == 6:
        psi =   f" {w[1, 0]                * w[0, 0]:.{p}f} * (I1 - 3) \\\\\
                 + {w[1, 1]:.{p}f} * (e^{{  {w[0, 1]:.{p}f} * (I1 - 3)}} - 1)\\\\\
                 - {w[1, 2]:.{p}f} * ln(1 - {w[0, 2]:.{p}f} * (I1 - 3)) \\\\\
                  \
                 + {w[1, 3]                * w[0, 3]:.{p}f} * (I2 - 3) \\\\\
                 + {w[1, 4]:.{p}f} * (e^{{  {w[0, 4]:.{p}f} * (I2 - 3))}} - 1)\\\\\
                 - {w[1, 5]:.{p}f} * ln(1 - {w[0, 5]:.{p}f} *  (I2 - 3)) \\\\"

    elif terms == 12:
        psi =   f" {w[1, 0]                * w[0, 0]:.{p}f} * (I1 - 3) \\\\\
                 + {w[1, 1]:.{p}f} * (e^{{  {w[0, 1]:.{p}f} * (I1 - 3)}} - 1)\\\\\
                 - {w[1, 2]:.{p}f} * ln(1 - {w[0, 2]:.{p}f} * (I1 - 3)) \\\\\
                 + {w[1, 3]                * w[0, 3]:.{p}f} * (I1 - 3) ^ 2 \\\\\
                 + {w[1, 4]:.{p}f} * (e^{{  {w[0, 4]:.{p}f} * (I1 - 3) ^ 2}} - 1)\\\\\
                 - {w[1, 5]:.{p}f} * ln(1 - {w[0, 5]:.{p}f} * (I1 - 3) ^ 2) \\\\\
                  \
                 + {w[1, 6]                * w[0, 6]:.{p}f} * (I2 - 3) \\\\\
                 + {w[1, 7]:.{p}f} * (e^{{  {w[0, 7]:.{p}f} * (I2 - 3))}} - 1)\\\\\
                 - {w[1, 8]:.{p}f} * ln(1 - {w[0, 8]:.{p}f} *  (I2 - 3)) \\\\\
                 + {w[1, 9]                * w[0, 9]:.{p}f} * (I2 - 3) ^ 2 \\\\\
                 + {w[1, 10]:.{p}f} * (e^{{ {w[0, 10]:.{p}f} *(I2 - 3) ^ 2)}} - 1)\\\\\
                 - {w[1, 11]:.{p}f} * ln(1 -{w[0, 11]:.{p}f} * (I2 - 3) ^ 2)\\\\"

    elif terms == 8:
        psi =   f" {w[1, 0]                * w[0, 0]:.{p}f} * (I1 - 3) \\\\\
                 + {w[1, 1]:.{p}f} * (e^{{  {w[0, 1]:.{p}f} * (I1 - 3)}} - 1)\\\\\
                 + {w[1, 2]                * w[0, 2]:.{p}f} * (I1 - 3) ^ 2 \\\\\
                 + {w[1, 3]:.{p}f} * (e^{{  {w[0, 3]:.{p}f} * (I1 - 3) ^ 2}} - 1)\\\\\
                  \
                 + {w[1, 4]                * w[0, 4]:.{p}f} * (I2 - 3) \\\\\
                 + {w[1, 5]:.{p}f} * (e^{{  {w[0, 5]:.{p}f} * (I2 - 3))}} - 1)\\\\\
                 + {w[1, 6]                * w[0, 6]:.{p}f} * (I2 - 3) ^ 2 \\\\\
                 + {w[1, 7]:.{p}f} * (e^{{ {w[0,  7]:.{p}f} *(I2 - 3) ^ 2)}} - 1)\\\\"

    elif terms == 16:
        psi =     f"{w[1, 0]               * w[0, 0]:.{p}f} * (I1 - 3) \\\\\
                  + {w[1, 1]:.{p}f} * (e^{{ {w[0, 1]:.{p}f} * (I1 - 3)}} - 1)\\\\\
                  + {w[1, 2]               * w[0, 2]:.{p}f} * (I1 - 3) ^ 2 \\\\\
                  + {w[1, 3]:.{p}f} * (e^{{ {w[0, 3]:.{p}f} * (I1 - 3) ^ 2}} - 1)\\\\\
                   \
                  + {w[1, 4]               * w[0, 4]:.{p}f} * (I2 - 3) \\\\\
                  + {w[1, 5]:.{p}f} * (e^{{ {w[0, 5]:.{p}f} * (I2 - 3))}} - 1)\\\\\
                  + {w[1, 6]               * w[0, 6]:.{p}f} * (I2 - 3) ^ 2 \\\\\
                  + {w[1, 7]:.{p}f} * (e^{{ {w[0, 7]:.{p}f} * (I2 - 3) ^ 2)}} - 1) \\\\\
                   \
                  + {w[1, 8]               * w[0, 8]:.{p}f} * (I4 - 3) \\\\ \
                  + {w[1, 9]:.{p}f} * (e^{{ {w[0, 9]:.{p}f} * (I4 - 3)}} - 1)\\\\ \
                  + {w[1, 10]              * w[0, 10]:.{p}f} * (I4 - 3) ^ 2 \\\\ \
                  + {w[1, 11]:.{p}f} * (e^{{{w[0, 11]:.{p}f} * (I4 - 3) ^ 2}} - 1)\\\\ \
                   \
                  + {w[1, 12]              * w[0, 12]:.{p}f} * (I5 - 3) \\\\ \
                  + {w[1, 13]:.{p}f} * (e^{{{w[0, 13]:.{p}f} * (I5 - 3))}} - 1)\\\\ \
                  + {w[1, 14]              * w[0, 14]:.{p}f} * (I5 - 3) ^ 2 \\\\ \
                  + {w[1, 15]:.{p}f} * (e^{{{w[0, 15]:.{p}f} * (I5 - 3) ^ 2)}} - 1)\\\\ "

    elif terms == 24:
        psi =     (f"{w[1, 0]                  * w[0, 0]:.{p}f} * (I1 - 3) \\\\\
                  + {w[1, 1]:.{p}f} * (e^{{     {w[0, 1]:.{p}f} * (I1 - 3)}} - 1)\\\\\
                  - {w[1, 2]:.{p}f} * (log{{1 - {w[0, 2]:.{p}f} * (I1 - 3)}})\\\\\
                  + {w[1, 3]                   * w[0, 3]:.{p}f} * (I1 - 3) ^ 2 \\\\\
                  + {w[1, 4]:.{p}f} * (e^{{     {w[0, 4]:.{p}f} * (I1 - 3) ^ 2}} - 1)\\\\\
                  - {w[1, 5]:.{p}f} * (log{{1 - {w[0, 5]:.{p}f} * (I1 - 3)^2}})\\\\\
                   \
                  + {w[1, 6]                  * w[0, 6]:.{p}f} * (I2 - 3) \\\\\
                  + {w[1, 7]:.{p}f} * (e^{{     {w[0, 7]:.{p}f} * (I2 - 3)}} - 1)\\\\\
                  - {w[1, 8]:.{p}f} * (log{{1 - {w[0, 8]:.{p}f} * (I2 - 3)}})\\\\\
                  + {w[1, 9]                   * w[0, 9]:.{p}f} * (I2 - 3) ^ 2 \\\\\
                  + {w[1, 10]:.{p}f} * (e^{{     {w[0, 10]:.{p}f} * (I2 - 3) ^ 2}} - 1)\\\\\
                  - {w[1, 11]:.{p}f} * (log{{1 - {w[0, 11]:.{p}f} * (I2 - 3)^2}})\\\\\
                   \
                  + {w[1, 12]                  * w[0, 12]:.{p}f} * (I4 - 1) \\\\\
                  + {w[1, 13]:.{p}f} * (e^{{     {w[0, 13]:.{p}f} * (I4 - 1)}} - 1)\\\\\
                  - {w[1, 14]:.{p}f} * (log{{1 - {w[0, 14]:.{p}f} * (I4 - 1)}})\\\\\
                  + {w[1, 15]                   * w[0, 15]:.{p}f} * (I4 - 1) ^ 2 \\\\\
                  + {w[1, 16]:.{p}f} * (e^{{     {w[0, 16]:.{p}f} * (I4 - 1) ^ 2}} - 1)\\\\\
                  - {w[1, 17]:.{p}f} * (log{{1 - {w[0, 17]:.{p}f} * (I4 - 1)^2}})\\\\\
                   \
                  +{w[1, 18]                  * w[0, 18]:.{p}f} * (I5 - 1) \\\\\
                  + {w[1, 19]:.{p}f} * (e^{{     {w[0, 19]:.{p}f} * (I5 - 1)}} - 1)\\\\\
                  - {w[1, 20]:.{p}f} * (log{{1 - {w[0, 20]:.{p}f} * (I5 - 1)}})\\\\\
                  + {w[1, 21]                   * w[0, 21]:.{p}f} * (I5 - 1) ^ 2 \\\\\
                  + {w[1, 22]:.{p}f} * (e^{{     {w[0, 22]:.{p}f} * (I5 - 1) ^ 2}} - 1)\\\\\
                  - {w[1, 23]:.{p}f} * (log{{1 - {w[0, 23]:.{p}f} * (I5 - 1)^2}})\\\\\ ")
    return psi


def stress_calc_ux(inputs):

    dPsidI1, dPsidI2, Stretch1, Stretch2 = inputs

    one = torch.tensor(1.0, dtype=torch.float32)
    two = torch.tensor(2.0, dtype=torch.float32)
    three = torch.tensor(3.0, dtype=torch.float32)
    four = torch.tensor(4.0, dtype=torch.float32)

    P11 = two * (dPsidI1 + one / Stretch1 * dPsidI2) * (Stretch1 - 1 / Stretch1 ** 2)
    return torch.cat((P11, P11), dim=1)
    # return P11


def stress_calc_bx(inputs, iso=False):

    if iso:
        dPsidI1, dPsidI2, Stretch1, Stretch2 = inputs
        dPsidI4, dPsidI5, al = 0, 0, torch.tensor(0)
    else:
        dPsidI1, dPsidI2, dPsidI4, dPsidI5, Stretch1, Stretch2, al = inputs

    one = torch.tensor(1.0, dtype=torch.float32)
    two = torch.tensor(2.0, dtype=torch.float32)
    three = torch.tensor(3.0, dtype=torch.float32)
    four = torch.tensor(4.0, dtype=torch.float32)

    # minus = two * (dPsidI1 * 1 / (Stretch ** 2) + dPsidI2 * 1 / (Stretch ** 3))
    # stress = two * (dPsidI1 * Stretch + dPsidI2 * one) - minus
    first_11 = (Stretch1 - one / (Stretch1 ** two * Stretch2 ** two))
    second_11 = (Stretch1 * Stretch2 ** two + one / (Stretch1 * Stretch2 ** two) - one / (Stretch1 ** two) - one / (
                Stretch2 ** two))
    fourth_11 = Stretch1 * torch.cos(al) ** two
    fifth_11 = Stretch1 ** three * torch.cos(al) ** two

    first_22 = (Stretch2 - one / (Stretch1 ** two * Stretch2 ** two))
    second_22 = (Stretch1 ** two * Stretch2 + one / (Stretch1 ** two * Stretch2) - one / (Stretch1 ** two) - one / (
                Stretch2 ** two))
    fourth_22 = Stretch2 * torch.sin(al) ** two
    fifth_22 = Stretch2 ** three * torch.sin(al) ** two

    P11 = two * (first_11 * dPsidI1 + second_11 * dPsidI2 + fourth_11 * dPsidI4 + two * fifth_11 * dPsidI5)
    P22 = two * (first_22 * dPsidI1 + second_22 * dPsidI2 + fourth_22 * dPsidI4 + two * fifth_22 * dPsidI5)
    return torch.cat((P11, P22), dim=1)


def compute_invariants(Stretch_x, Stretch_y, exp_type):
    """
    Вычисляет инварианты для каждого образца в батче в зависимости от типа эксперимента.

    Для exp_type == 1000 используются альтернативные формулы, для остальных — классические.

    Параметры:
      Stretch_x, Stretch_y: тензоры формы (batch_size, 1) с растяжениями.
      exp_type: тензор формы (batch_size,) с типами эксперимента.

    Возвращает:
      I1, I2: тензоры инвариантов формы (batch_size, 1).
    """
    batch_size = Stretch_x.shape[0]
    device = Stretch_x.device
    I1 = torch.zeros((batch_size, 1), device=device)
    I2 = torch.zeros((batch_size, 1), device=device)

    # Маски для разных типов эксперимента
    mask_1000 = torch.tensor([x == "uni" for x in exp_type])
    mask_other = torch.tensor([x != "uni" for x in exp_type])

    # Для образцов с exp_type == 1000 – альтернативное вычисление инвариантов
    if mask_1000.sum() > 0:
        # Пример: можно использовать более простые формулы
        I1[mask_1000] = Stretch_x[mask_1000] ** 2 + 2 / Stretch_x[mask_1000]
        I2[mask_1000] = 2 * Stretch_x[mask_1000] + 1 / Stretch_y[mask_1000] ** 2

    # Для остальных типов эксперимента – классические формулы
    if mask_other.sum() > 0:
        # Вычисляем дополнительный параметр деформации
        Stretch_z = 1 / (Stretch_x[mask_other] * Stretch_y[mask_other])
        I1[mask_other] = Stretch_x[mask_other] ** 2 + Stretch_y[mask_other] ** 2 + Stretch_z ** 2
        I2[mask_other] = (Stretch_x[mask_other] ** 2) * (Stretch_y[mask_other] ** 2) \
                         + 1 / (Stretch_x[mask_other] ** 2) + 1 / (Stretch_y[mask_other] ** 2)
    return I1, I2


def compute_stress(dWI1_BT, dWdI2_BT, Stretch_x, Stretch_y, exp_type):
    """
    Вычисляет итоговый тензор напряжений для батча с разными типами экспериментов.

    Для образцов с exp_type == 1000 используется одноканальный вывод,
    для остальных – двухканальный.

    Параметры:
      dWI1_BT, dWdI2_BT: производные потенциала по инвариантам I1 и I2.
      Stretch_x, Stretch_y: тензоры растяжений, форма (batch_size, 1).
      exp_type: тензор типов эксперимента, форма (batch_size,).

    Возвращает:
      stress_out: тензор формы (2, batch_size), где для exp_type==1000 заполнен только первый канал.
    """
    mask_1000 = torch.tensor([x == "uni" for x in exp_type])
    mask_other = torch.tensor([x != "uni" for x in exp_type])

    batch_size = Stretch_x.shape[0]

    # Инициализируем итоговый тензор для двух каналов
    stress_out = torch.zeros((2, batch_size))

    # Обработка для exp_type == 1000 (одноканальный вывод)
    if mask_1000.sum() > 0:
        dWI1_1000 = dWI1_BT[mask_1000]
        dWdI2_1000 = dWdI2_BT[mask_1000]
        Stretch_x_1000 = Stretch_x[mask_1000]
        Stretch_y_1000 = Stretch_y[mask_1000]
        # stress_calc_bx возвращает тензор формы (2, N_1000)
        stress_1000 = stress_calc_ux((dWI1_1000, dWdI2_1000, Stretch_x_1000, Stretch_y_1000))
        stress_out[0, mask_1000] = stress_1000[:, 0]

    # Обработка для остальных типов эксперимента (двухканальный вывод)
    if mask_other.sum() > 0:
        dWI1_other = dWI1_BT[mask_other]
        dWdI2_other = dWdI2_BT[mask_other]
        Stretch_x_other = Stretch_x[mask_other]
        Stretch_y_other = Stretch_y[mask_other]
        # stress_calc_bx возвращает тензор формы (2, N_other)
        stress_other = stress_calc_bx((dWI1_other, dWdI2_other, Stretch_x_other, Stretch_y_other), iso=True)
        stress_out[:, mask_other] = stress_other.T

    return stress_out.T
