import os
import tempfile

import numpy as np
# from fastapi import BackgroundTasks

# Импорты для работы с CANN моделями (опциональные)
try:
    from ..models.CANN import ModelArchitecture_I5, ModelArchitecture_I2
except ImportError:
    # Если модули недоступны, продолжаем работу без CANN функционала
    ModelArchitecture_I5 = None
    ModelArchitecture_I2 = None

# from app.modules.isotropic.solver import IsotropicModelType


class EnergyInfo:

    # @staticmethod
    # async def download_energy(energy_text: str, background_tasks: BackgroundTasks) -> str:
    #     with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix=".energy", encoding="utf-8") as temp_file:
    #         temp_file.write(energy_text)
    #         temp_file_path = temp_file.name

    #     background_tasks.add_task(os.remove, temp_file_path)

    #     return temp_file_path

    # @staticmethod
    # def energy_text(name: str, params: np.ndarray) -> str:
    #     # TODO Переписать на фабрику моделей
    #     """
    #     Возвращает текст .energy в стиле FEBio-Calc.
    #     3-D инварианты (J=1):
    #         I3d_1 = I[1] + 1/I[2]
    #         I3d_2 = I[2] + I[1]/I[2]
    #     """
    #     model_name_internal = IsotropicModelType(name)
    #     if model_name_internal is None:
    #         raise ValueError(f"Unknown model name alias: {name}")
    #     hdr = f"# Auto-generated .energy for {name}\n\n"

    #     if model_name_internal == "NeoHookean":
    #         (mu,) = params
    #         return (
    #                 hdr +
    #                 f'Var mu : "f:mu" = {mu:.8g};  # [MPa]\n\n'
    #                 "Let I3d_1 = I[1] + 1/I[2];\n"
    #                 "Potential = mu/2 * (I3d_1 - 3);\n"
    #         )

    #     if model_name_internal == "MooneyRivlin":
    #         c1, c2 = params
    #         return (
    #                 hdr +
    #                 f'Var c1 : "f:c1" = {c1:.8g}, '
    #                 f'c2 : "f:c2" = {c2:.8g};  # [MPa]\n\n'
    #                 "Let I3d_1 = I[1] + 1/I[2], I3d_2 = I[2] + I[1]/I[2];\n"
    #                 "Potential = c1 * (I3d_1 - 3) + c2 * (I3d_2 - 3);\n"
    #         )

    #     if model_name_internal == "GeneralizedMooneyRivlin":
    #         C10, C01, C11, C20, C02 = params
    #         return (
    #                 hdr +
    #                 "Var C10=\"f:C10\"={:.8g}, C01=\"f:C01\"={:.8g}, "
    #                 "C11={:.8g}, C20={:.8g}, C02={:.8g};  # [MPa]\n\n"
    #                 "Let I3d_1 = I[1] + 1/I[2], I3d_2 = I[2] + I[1]/I[2];\n"
    #                 "Potential = C10*(I3d_1-3) + C01*(I3d_2-3) + "
    #                 "C11*(I3d_1-3)*(I3d_2-3) + C20*(I3d_1-3)^2 + "
    #                 "C02*(I3d_2-3)^2;\n".format(C10, C01, C11, C20, C02)
    #         )

    #     if model_name_internal == "Yeoh":
    #         c1, c2, c3 = params
    #         return (
    #                 hdr +
    #                 f'Var c1="f:c1"={c1:.8g}, c2={c2:.8g}, c3={c3:.8g};  # [MPa]\n\n'
    #                 "Let I3d_1 = I[1] + 1/I[2];\n"
    #                 "Potential = c1*(I3d_1-3) + c2*(I3d_1-3)^2 + "
    #                 "c3*(I3d_1-3)^3;\n"
    #         )

    #     if model_name_internal == "Beda":
    #         c1, c2, c3, K1, a, k, b = params
    #         return (
    #                 hdr +
    #                 "Var c1={:.8g}, c2={:.8g}, c3={:.8g}, "
    #                 "K1={:.8g}, a={:.8g}, k={:.8g}, b={:.8g};\n\n"
    #                 "Let I3d_1 = I[1] + 1/I[2], I3d_2 = I[2] + I[1]/I[2];\n"
    #                 "Potential = c1/a*(I3d_1-3)^a + c2*(I3d_2-3) + "
    #                 "c3/k*(I3d_1-3)^k + K1/b*(I3d_2-3)^b;\n".format(
    #                     c1, c2, c3, K1, a, k, b
    #                 )
    #         )

    #     if model_name_internal == "Gent":
    #         mu, Jm = params
    #         return (
    #                 hdr +
    #                 f'Var mu="f:mu"={mu:.8g}, Jm={Jm:.8g};  # [MPa], [-]\n\n'
    #                 "Let I3d_1 = I[1] + 1/I[2];\n"
    #                 "Potential = -mu*Jm/2*log(1 - (I3d_1-3)/Jm);\n"
    #         )

    #     if model_name_internal == "Carroll":
    #         A, B, C = params
    #         return (
    #                 hdr +
    #                 f'Var A={A:.8g}, B={B:.8g}, C={C:.8g};  # [MPa]\n\n'
    #                 "Let I3d_1 = I[1] + 1/I[2], I3d_2 = I[2] + I[1]/I[2];\n"
    #                 "Potential = A*I3d_1 + B*I3d_1^4 + C*sqrt(I3d_2);\n"
    #         )

    #     return hdr + "# Unknown model\nPotential = 0;\n"

    @staticmethod
    def energy_text_from_cann(model, model_name: str = "CANN_Model", force_isotropic: bool = False, precision_p: int = 3) -> str:
        """
        Генерирует текст .energy файла из обученной CANN модели.
        Только коэффициенты с |значением| >= 10^{-precision_p} попадают в вывод.
        """
        
        tol = 10.0 ** (-precision_p)
        
        # Получаем веса модели
        if not hasattr(model, 'potential_constants') or model.potential_constants is None:
            model.get_weights()
        
        weights = model.potential_constants  # shape: [2, N]
        
        # Заголовок
        hdr = f"# Auto-generated .energy for {model_name} (CANN)\n"
        hdr += f"# Model type: {model.__class__.__name__}\n"
        hdr += f"# Total terms (raw): {weights.shape[1]}\n"
        
        # Определяем тип модели и количество инвариантов
        is_anisotropic = "I5" in model.__class__.__name__ or hasattr(model, 'H_layer')
        
        # Применяем force_isotropic
        if force_isotropic and is_anisotropic:
            hdr += "# ВНИМАНИЕ: Анизотропная модель конвертирована в изотропную (I4, I5 игнорируются)\n"
            is_anisotropic = False  # переопределяем для дальнейшей логики
        
        hdr += f"# Threshold: |coef| >= 1e-{precision_p}\n\n"
        
        if is_anisotropic:
            # Анизотропная модель (I1, I2, I4, I5)
            invariants_section = (
                "# Инварианты для анизотропной модели\n"
                "Fiber f, s;  # направления волокон (пример)\n"
                "Let I3d_1 = I[1] + 1/I[2];  # I1 в 3D\n"
                "Let I3d_2 = I[2] + I[1]/I[2];  # I2 в 3D\n"
                "Let I4 = I[f];      # Структурный инвариант I4\n"
                "Let I5 = I[f,s];    # Структурный инвариант I5\n\n"
            )
            invariant_names = ["I3d_1", "I3d_2", "I4", "I5"]
            invariant_biases = [3, 3, 1, 1]  # стандартные смещения
        else:
            # Изотропная модель (только I1, I2)
            invariants_section = (
                "# Инварианты для изотропной модели\n"
                "Let I3d_1 = I[1] + 1/I[2];  # I1 в 3D\n"
                "Let I3d_2 = I[2] + I[1]/I[2];  # I2 в 3D\n\n"
            )
            invariant_names = ["I3d_1", "I3d_2"]
            invariant_biases = [3, 3]
        
        # --- Динамически определяем активации и степень полинома ----------------------
        # Берём первую инвариантную сеть, она задаёт порядок терминов
        if hasattr(model.Psi_model, "I1_net"):
            inv_net_ref = model.Psi_model.I1_net
        elif hasattr(model.Psi_model, "invariant_nets"):
            inv_net_ref = model.Psi_model.invariant_nets[0]
        else:
            raise RuntimeError("Не удалось найти инвариантную сеть в модели – неизвестная структура Psi_model")

        activation_functions = list(inv_net_ref.activation_functions)  # сохраняем порядок
        polynomial_degree = int(inv_net_ref.polynomial_degree)

        terms_per_invariant = len(activation_functions) * polynomial_degree

        # --------------------------------------------------------------------------------
        
        # Строим потенциал
        potential_terms = []
        var_declarations = []
        term_idx = 0
        
        # --- словарь констант ---------------------------------------------------------
        c_vals: dict[int, float] = {}
        k_vals: dict[int, float] = {}
        
        for inv_idx, (inv_name, bias) in enumerate(zip(invariant_names, invariant_biases)):
            # Для каждого инварианта: linear(I-b), exp(I-b), linear((I-b)^2), exp((I-b)^2)
            for poly_deg in range(1, polynomial_degree + 1):  # (I-b)^1, (I-b)^2, ...
                for activation in activation_functions:
                    if term_idx >= weights.shape[1]:
                        break
                    
                    w_inner = weights[0, term_idx].item()  # внутренний вес (w0)
                    w_final = weights[1, term_idx].item()  # внешний вес (w1)

                    coef_var = f"c{term_idx+1}"
                    inner_var = f"k{term_idx+1}"

                    # Пороговая фильтрация — решаем, включать ли термин
                    if activation == "linear":
                        coeff = w_inner * w_final
                        keep = abs(coeff) >= tol
                    else:
                        # для exp/ln требуем: амплитуда и множитель не малы
                        keep = (abs(w_final) >= tol) and (abs(w_inner) >= tol)

                    if keep:
                        # формируем Var и Potential
                        if activation == "linear":
                            var_declarations.append(f"{coef_var} = {coeff:.8g}")
                        else:
                            var_declarations.append(f"{coef_var} = {w_final:.8g}")
                            var_declarations.append(f"{inner_var} = {w_inner:.8g}")

                        power_str = "" if poly_deg == 1 else f"^{poly_deg}"
                        ref_str = f"({inv_name} - {bias}){power_str}"

                        if activation == "linear":
                            potential_terms.append(f"{coef_var} * {ref_str}")
                            c_vals[term_idx+1] = coeff
                            k_vals[term_idx+1] = w_inner
                        elif activation == "exp":
                            potential_terms.append(f"{coef_var} * (exp({inner_var} * {ref_str}) - 1)")
                            c_vals[term_idx+1] = w_final
                            k_vals[term_idx+1] = w_inner
                        elif activation == "ln":
                            potential_terms.append(f"{coef_var} * ln(1 - {inner_var} * {ref_str})")
                            c_vals[term_idx+1] = w_final
                            k_vals[term_idx+1] = w_inner
                        else:
                            potential_terms.append(f"{coef_var} * {ref_str}  # {activation}")
                            c_vals[term_idx+1] = w_final
                            k_vals[term_idx+1] = w_inner
                    
                    term_idx += 1
        
        # Создаем секцию переменных: только из отфильтрованных Var
        if var_declarations:
            variables_section = "Var " + ",\n    ".join(var_declarations) + ";  # [MPa]\n\n"
        else:
            variables_section = "Var ;  # нет значимых коэффициентов\n\n"
        
        # Собираем финальный потенциал
        if potential_terms:
            potential_section = "Potential = " + " + \n           ".join(potential_terms) + ";\n"
        else:
            potential_section = "Potential = 0;  # Все коэффициенты ниже порога\n"
        
        # Объединяем все части
        energy_text = hdr + variables_section + invariants_section + potential_section
        
        return energy_text

    @staticmethod
    def save_energy_from_cann(model, model_name: str, output_dir: str = ".", force_isotropic: bool = False, precision_p: int = 3) -> str:
        """
        Сохраняет .energy файл из обученной CANN модели в указанную директорию.
        Коэффициенты ниже порога 10^{-precision_p} отбрасываются.
        """
        import os
        from pathlib import Path
        
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        energy_text = EnergyInfo.energy_text_from_cann(model, model_name, force_isotropic, precision_p)
        
        safe_name = "".join(c for c in model_name if c.isalnum() or c in (' ', '-', '_')).rstrip()
        
        if force_isotropic and ("I5" in model.__class__.__name__ or hasattr(model, 'H_layer')):
            filename = f"{safe_name}_isotropic.energy"
        else:
            filename = f"{safe_name}.energy"
        
        file_path = os.path.join(output_dir, filename)
        
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(energy_text)
            
        return file_path

    @staticmethod
    async def download_energy_from_cann(model, model_name: str) -> str:
        """
        Создает временный .energy файл из обученной CANN модели для скачивания.
        """
        energy_text = EnergyInfo.energy_text_from_cann(model, model_name)
        
        # Создаем безопасное имя файла
        safe_name = "".join(c for c in model_name if c.isalnum() or c in (' ', '-', '_')).rstrip()
        
        with tempfile.NamedTemporaryFile(
            mode='w', 
            delete=False, 
            suffix=f"_{safe_name}.energy", 
            encoding="utf-8"
        ) as temp_file:
            temp_file.write(energy_text)
            temp_file_path = temp_file.name

        # Возвращаем путь; очистка на вызывающей стороне
        return temp_file_path

"""
Пример использования:

# Загрузка обученной модели
from src.CANN_torch.models.CANN import ModelArchitecture_I5
import torch

# Создание или загрузка модели
model = ModelArchitecture_I5(...)  # инициализация модели
model.load_state_dict(torch.load("path/to/model.pth"))  # загрузка весов

# Генерация .energy файла
from src.CANN_torch.utils.energy_info import EnergyInfo

energy_text = EnergyInfo.energy_text_from_cann(model, "My_CANN_Model")
print(energy_text)

# Сохранение в файл
with open("my_model.energy", "w", encoding="utf-8") as f:
    f.write(energy_text)

# Для использования в FastAPI (асинхронно)
async def export_cann_energy(model, background_tasks):
    file_path = await EnergyInfo.download_energy_from_cann(model, "CANN_Export", background_tasks)
    return file_path

Результат будет выглядеть примерно так:

# Auto-generated .energy for My_CANN_Model (CANN)
# Model type: ModelArchitecture_I5
# Total terms: 16

Var c1 = 0.0521, c2 = 0.0832, c3 = 0.0042, c4 = 0.1205, ...;  # [MPa]

# Инварианты для анизотропной модели
Fiber f, s;  # направления волокон (пример)
Let I3d_1 = I[1] + 1/I[2];  # I1 в 3D
Let I3d_2 = I[2] + I[1]/I[2];  # I2 в 3D
Let I4 = I[f];      # Структурный инвариант I4
Let I5 = I[f,s];    # Структурный инвариант I5

Potential = c1 * (I3d_1 - 3) + 
           c2 * (exp(0.597 * (I3d_1 - 3)) - 1) + 
           c3 * (I3d_1 - 3)^2 + 
           c4 * (exp(0.788 * (I3d_1 - 3)^2) - 1) + 
           c5 * (I3d_2 - 3) + 
           ...
           c16 * (exp(0.923 * (I5 - 1)^2) - 1);
"""
