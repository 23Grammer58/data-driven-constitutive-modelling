import sympy as sp


def get_potential_formula_2term(model):
    raw_params = model.get_potential()
    mid = len(raw_params) // 2
    first_half = raw_params[:mid]
    second_half = raw_params[mid:]
    coefficients = [first_half[i] * second_half[i] for i in range(mid)]

    # Инициализация символов SymPy
    I1, I2 = sp.symbols('I1 I2')

    # Проверка на количество коэффициентов
    # if len(coefficients) < 4:
    #     raise ValueError("Для формулы требуется как минимум 4 коэффициента")

    # Подставляем коэффициенты в формулу
    psi = (coefficients[0] * (I1 - 3) + coefficients[2] * (I2 - 3) +
           coefficients[1] * (I1 - 3) ** 2 + coefficients[3] * (I2 - 3) ** 2)

    # Вывод формулы
    sp.pprint(psi)


def get_potential_formula_4term(model):
    raw_params = model.get_potential()
    # print(raw_params)
    mid = len(raw_params) // 2
    first_half = raw_params[:mid]
    second_half = raw_params[mid:]
    coefficients = [first_half[i] * second_half[i] for i in range(mid)]

    # Инициализация символов SymPy
    I1, I2 = sp.symbols('I1 I2')

    # Проверка на количество коэффициентов
    # if len(coefficients) < 4:
    #     raise ValueError("Для формулы требуется как минимум 4 коэффициента")

    # Подставляем коэффициенты в формулу
    psi = (coefficients[0] * (I1 - 3)
           + coefficients[2] * (I2 - 3)
           + coefficients[1] * (I1 - 3) ** 2
           + coefficients[3] * (I2 - 3) ** 2
           + coefficients[4] * sp.exp(I1 - 3)
           + coefficients[6] * sp.exp(I2 - 3)
           + coefficients[5] * sp.exp(I1 - 3) ** 2
           + coefficients[7] * sp.exp(I2 - 3) ** 2)

    # Вывод формулы
    sp.pprint(psi)


def get_potential_formula_6term(model):
    coefficients = model.get_potential()

    # Инициализация символов SymPy
    I1, I2 = sp.symbols('I1 I2')

    # Подставляем коэффициенты в формулу
    psi = (  coefficients[0] * (I1 - 3)
           + coefficients[2] * (I2 - 3)
           + coefficients[1] * (I1 - 3) ** 2
           + coefficients[3] * (I2 - 3) ** 2
           + coefficients[4] * sp.exp(I1 - 3)
           + coefficients[6] * sp.exp(I2 - 3)
           + coefficients[5] * sp.exp(I1 - 3) ** 2
           + coefficients[7] * sp.exp(I2 - 3) ** 2
           + coefficients[9] * sp.exp(I1 - 3) ** 2
           + coefficients[11] * sp.exp(I2 - 3) ** 2

    )

    # Вывод формулы
    sp.pprint(psi)


