# CLANN - Constitutive Law Artificial Neural Network

Проект для моделирования конститутивных зависимостей материалов с использованием искусственных нейронных сетей.

## Описание

CLANN (Constitutive Law Artificial Neural Network) - это инструмент для создания и обучения нейронных сетей, моделирующих конститутивные уравнения материалов. Проект включает в себя различные архитектуры нейронных сетей и инструменты для анализа данных.

## Структура проекта

```
clann/
├── data/                    # Экспериментальные данные
│   └── xi_0_1_stretch1_01_txt.csv
├── experiments/             # Результаты экспериментов
│   ├── anchor_full/
│   └── anchor_test/
│       ├── anchor_test_hessian.onnx
│       ├── anchor_test.onnx
│       ├── anchor_test.pth
│       └── stress_strain_*.png
├── old/                     # Старые версии кода
│   ├── analytical.py
│   ├── clann_v2.py
│   ├── clann.py
│   └── train_clann.py
├── venv/                    # Виртуальное окружение
├── clann_shear.pth         # Обученная модель
├── hessian_wrapper.onnx    # ONNX модель
├── train_clann_v2.py       # Основной скрипт обучения
├── requirements.txt         # Зависимости
├── run_tests.bat           # Скрипт для запуска тестов
└── clann_usage.md          # Документация по использованию
```

## Установка

1. Клонируйте репозиторий:
```bash
git clone https://github.com/23Grammer58/data-driven-constitutive-modelling.git
cd data-driven-constitutive-modelling
```

2. Создайте виртуальное окружение:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# или
venv\Scripts\activate     # Windows
```

3. Установите зависимости:
```bash
pip install -r requirements.txt
```

## Использование

Основной скрипт для обучения модели:
```bash
python train_clann_v2.py
```

Для запуска тестов:
```bash
run_tests.bat
```

## Зависимости

Основные зависимости указаны в `requirements.txt`:
- PyTorch
- NumPy
- Matplotlib
- Pandas
- ONNX

## Лицензия

MIT License

## Автор

Проект является частью исследований в области моделирования конститутивных зависимостей материалов.
