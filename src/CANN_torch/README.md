# CANN_torch

Библиотека для реализации Constitutive Artificial Neural Networks (CANN) с использованием PyTorch.

## Описание

CANN_torch - это библиотека для создания и обучения нейронных сетей, которые моделируют конститутивные уравнения материалов. Библиотека реализует различные архитектуры CANN, включая:

- Базовые CANN модели
- CNN-модификации CANN
- Различные функции потенциала

## Установка

```bash
pip install -e .
```

## Основные компоненты

### Модели
- `StrainEnergyCANN` - базовая CANN модель
- `StrainEnergyCANN_Ani` - анизотропная CANN модель
- Различные реализации CNN-модификаций

### Утилиты
- Загрузка и предобработка данных
- Визуализация результатов
- Валидация моделей

## Пример использования

```python
from CANN_torch.models import StrainEnergyCANN
from CANN_torch.core import Trainer
from CANN_torch.utils import ExcelDataset

# Создание модели
model = StrainEnergyCANN()

# Создание тренера
trainer = Trainer(
    model=model,
    experiment_name="test_experiment",
    learning_rate=0.001
)

# Загрузка данных
dataset = ExcelDataset("path_to_data.xlsx")
train_loader = DataLoader(dataset, batch_size=32)

# Обучение модели
trainer.train(train_loader)
```

## Лицензия

MIT 