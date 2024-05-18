# data-driven-constitutive-modelling

Автоматический поиск гиперупругой изотропной феноменологической модели биоматериала при помощи архитектуры [CANN](https://github.com/LivingMatterLab/CANN) реализованная на фреймворке PyTorch.

## Установка
```commandline
git clone https://github.com/23Grammer58/data-driven-constitutive-modelling
cd data-driven-constitutive-modelling
pip install -e .
```

## Описание функционала класса `Trainer`

Класс `Trainer` предназначен для обучения моделей Constitutive Artificial Neural Networks (CANNs).

### Аргументы

- `experiment_name` (str): Название эксперимента. По умолчанию "test".
- `model` (nn.Module): Архитектура модели для обучения. По умолчанию `StrainEnergyCANN_C`.
- `path_to_save_weights` (str): Путь для сохранения весов модели.
- `epochs` (int): Количество эпох обучения. По умолчанию 100.
- `learning_rate` (float): Скорость обучения. По умолчанию 0.001.
- `plot_valid` (bool): Отрисовка ошибки на валидационном датасете. По умолчанию False.
- `l1_reg_coeff` (float): Коэффициент регуляризации L1. По умолчанию 0.001.
- `l2_reg_coeff` (float): Коэффициент регуляризации L2. По умолчанию 0.001.
- `checkpoint` (str): Путь до весов модели, с которыми модель инициализируется. По умолчанию 1.
- `device` (str): Устройство для выполнения вычислений (cpu или cuda). По умолчанию "cpu".

## Литература 
Linka, Kevin, Sarah R. St Pierre, and Ellen Kuhl. "Automated model discovery for human brain using Constitutive Artificial Neural Networks." Acta Biomaterialia 160 (2023): 134-151.


## License
This project is licensed under the MIT License.

