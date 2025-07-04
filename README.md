# CANN_torch

Библиотека для реализации Constitutive Artificial Neural Networks (CANN) с использованием PyTorch для задач идентификации конститутивных зависимостей в механике деформируемого твёрдого тела.

## Описание

CANN_torch — современная библиотека для создания и обучения нейронных сетей, моделирующих конститутивные уравнения материалов. Поддерживает различные архитектуры CANN с физически обоснованными ограничениями и автоматической генерацией strain-energy функций.

### Основные возможности

- 🧠 **CANN модели**: ModelArchitecture_I5, ModelArchitecture_I2 с инвариантными сетями
- 🔬 **Физические ограничения**: объективность, поликонвексность, монотонность
- 📊 **Унифицированный пайплайн**: от данных до LaTeX-формул потенциала
- ⚡ **Микросервисная архитектура**: готов к интеграции в бекенд
- 📈 **Автоматическая визуализация**: R²-метрики, графики предсказаний
- 🎯 **Weighted training**: поддержка весов для разных типов экспериментов

## Установка

```bash
pip install -e .
```

## Быстрый старт

### CLI интерфейс (рекомендуется)

```bash
# Базовое обучение
python -m CANN_torch.pipeline.CANN_pipeline \
       --data_dir data/GoreTex/2/DIC/3 \
       --epochs 50 --batch_size 16

# С настройкой весов для разных экспериментов
python -m CANN_torch.pipeline.CANN_pipeline \
       --data_dir data/experiments \
       --train "100_100,050_100" \
       --test all \
       --weights "uni=2.0,100_100=1.0,050_100=0.5" \
       --epochs 100
```

### Программный интерфейс

```python
from CANN_torch.pipeline import TrainingConfig, train_and_evaluate

# Конфигурация обучения
cfg = TrainingConfig(
    experiment_dir="data/GoreTex/2/DIC/3",
    train_protocols=["100_100", "050_100"],
    test_protocols="all",
    epochs=50,
    batch_size=16,
    weights_map={"uni": 2.0, "100_100": 1.0}
)

# Запуск полного цикла
results = train_and_evaluate(cfg)
```

## Архитектура библиотеки

```
CANN_torch/
├── core/                    # Основные компоненты
│   └── trainer.py          # Класс обучения с weighted loss
├── models/                  # Архитектуры моделей
│   ├── CANN_gpt.py         # ModelArchitecture_I5, I2
│   └── CNN.py              # CNN-модификации
├── pipeline/                # Унифицированный пайплайн
│   └── CANN_pipeline.py    # Полный цикл: данные→обучение→результаты
└── utils/                   # Утилиты
    ├── dataload.py         # Загрузка экспериментальных данных
    ├── potential_zoo.py    # Расчёт напряжений и инвариантов
    └── visualisation.py    # Построение графиков
```

## Поддерживаемые модели

### ModelArchitecture_I5
Полная анизотропная модель с инвариантами I₁, I₂, I₄, I₅:
- **Входы**: λₓ, λᵧ (plant stretches)
- **Выходы**: P₁₁, P₂₂ (Piola-Kirchhoff stress)
- **Инварианты**: 4 × (linear + exp) × (I-3, (I-3)²) = 16 терминов

### ModelArchitecture_I2  
Изотропная модель с инвариантами I₁, I₂:
- **Входы**: λₓ, λᵧ + experiment_type
- **Выходы**: P₁₁, P₂₂
- **Инварианты**: 2 × (linear + exp) × (I-3, (I-3)²) = 8 терминов

## Формат данных

### CSV структура
```csv
# Xlam, PX, Ylam, PY, experiment_type
1.1, 0.5, 1.0, 0.2, 100_100
1.2, 0.8, 1.0, 0.2, 100_100
1.0, 0.0, 1.1, 0.5, uni
```

### Типы экспериментов
- `uni` — одноосное растяжение
- `100_100` — равноосное двухосное
- `100_075`, `075_100` — неравноосное двухосное
- Пользовательские типы

## Результаты обучения

После завершения в `results/run_YYYYMMDD_HHMMSS/` создаются:

- 📈 **Графики**: `plot_all_x.png`, `plot_all_y.png`
- 📊 **Метрики**: `metrics.csv` (R² по протоколам)
- 📋 **Сводка**: `summary.csv` (средние R², константы модели)
- 🧠 **Веса**: `best_model.pth`

### Автоматическая генерация потенциала

```latex
ψ = 0.291 * (I₁ - 3) + 0.195 * (e^{0.597 * (I₁ - 3)} - 1) + 
    0.002 * (I₁ - 3)² + 0.306 * (e^{0.788 * (I₁ - 3)²} - 1) + 
    ...
```

## Расширенные возможности

### Weighted Training
```python
# Веса-штрафы для разных типов экспериментов
weights_map = {
    "uni": 2.0,        # уни-эксперименты важнее
    "100_100": 1.0,    # базовый вес
    "complex": 0.3     # сложные эксперименты менее значимы
}
```

### Валидация моделей
```python
from CANN_torch.utils import compute_stress, compute_invariants

# Ручная проверка предсказаний
I1, I2 = compute_invariants(lambda_x, lambda_y, exp_type)
stress = compute_stress(dW_dI1, dW_dI2, lambda_x, lambda_y, exp_type)
```

## Интеграция в микросервисы

Библиотека спроектирована для лёгкой интеграции:

```python
# FastAPI пример
from fastapi import FastAPI
from CANN_torch.pipeline import TrainingConfig, train_and_evaluate

app = FastAPI()

@app.post("/train")
async def train_model(config: TrainingConfig):
    results = train_and_evaluate(config)
    return {"run_dir": results["run_dir"], "metrics": results["metrics"]}
```

## Лицензия

MIT License — свободное использование в исследованиях и коммерческих проектах.

