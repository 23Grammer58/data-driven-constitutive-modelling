# Руководство по использованию CLANN-интерполяторов

> Версия: ψ(ξ) → ∂ψ/∂ξ (стрессы) + ∂²ψ/∂ξ² (гессиан)
> 
> Этот документ описывает:
> • как собрать динамическую библиотеку с ONNX-Runtime;
> • какие функции она экспортирует;
> • какой формат данных ожидает/возвращает каждая функция;
> • как подключить библиотеку к FE-симулятору и Ньюто́новскому решателю.

---

## 1. Результаты, которые нужны FE-коду

| Обозначение | Математика                    | Требует FE | Размер |
|-------------|------------------------------|-----------|--------|
| `grad`      | ∂ψ/∂ξ (вектор)               | да (стрессы) | 3      |
| `hess`      | ∂²ψ/∂ξ² (полная 3×3 матрица) | да (матрица жёсткости, Ньютон) | 9 |

---

## 2. Экспортируемые функции

| Имя C-API          | Что вычисляет                 | Выход | Примечание |
|--------------------|-------------------------------|-------|------------|
| `run_gradient`     | вектор `grad`                 | 3 × float32 | основной отклик (стрессы) |
| `run_hessian`      | матрица `hess` row-major      | 9 × float32 | для Ньюто́на |
| `set_model_paths`  | установить пути к ONNX-файлам | —     | см. § 5     |

Для обратной совместимости доступны синонимы (не рекомендуется использовать дальше):
```
run_inference       → run_gradient
run_inference_hess  → run_hessian
run_hessian         → run_hessian   (оставлено)
```

---

## 3. Компиляция библиотеки

```bash
cd onnxruntime/nn_response/
chmod +x build_test.sh
./build_test.sh      # создаст libinference.so / inference.dll
```

После сборки убедитесь, что `nm -D libinference.so` (или `dumpbin /exports` под Windows) показывает функции из § 2.

---

## 4. Формат входа/выхода

*Везде используется float32; FE-код обязан при необходимости преобразовать в `double`.*

### 4.1 Вход
```
xi[0] = ξ₀ = ln u₁₁
xi[1] = ξ₁ = ln u₂₂
xi[2] = ξ₂ = u₁₂ / u₁₁
```

### 4.2 Выходы

| Функция        | Порядок значений                                                   |
|----------------|--------------------------------------------------------------------|
| `run_gradient` | `[g0, g1, g2] = [∂ψ/∂ξ₀, ∂ψ/∂ξ₁, ∂ψ/∂ξ₂]`                          |
| `run_hessian`  | `row-major {r00, r01, r02, r10, r11, r12, r20, r21, r22}`          |
| `run_hessian6` | `{r00, r11, r22, r10, r02, r12}` – порядок, требуемый LinearElasticResponse |

Матрица из `run_hessian` симметрична, но возвращается полной, чтобы исключить неоднозначность.

---

## 5. Настройка ONNX-моделей

### 5.1 Переменные окружения
```bash
export ONNX_RESPONSE_MODEL="/abs/path/psi_grad.onnx"   # для grad
export ONNX_HESSIAN_MODEL="/abs/path/psi_hess.onnx"    # для hess/hess6
```

### 5.2 Программно
```cpp
set_model_paths("psi_grad.onnx", "psi_hess.onnx");
```
Если не задано — используются значения по умолчанию
`experiments/onnx/log_xi.onnx` и `experiments/onnx/log_xi_hessian.onnx`.

---

## 6. Интеграция в FE-код

1. Загрузить `libinference.so`.
2. Получить указатели на функции § 2 (`dlsym` / `GetProcAddress`).
3. При вычислении напряжений использовать `run_gradient`.
4. Для матрицы жёсткости Ньюто́на использовать `run_hessian` (9 чисел → 3×3 `double`).
5. При желании ускорить/упростить SPD-проверку — использовать `run_hessian6`.

---

## 7. Флаги командной строки для обучения

### 7.1 Основные параметры

| Флаг | Тип | По умолчанию | Описание |
|------|-----|--------------|----------|
| `--csv` | str | `"data/FS_pairs.csv"` | Путь к CSV файлу с данными (F/C тензоры и PK2 напряжения) |
| `--xi_stretch_dir` | str | `None` | Директория с xi_stretch файлами (альтернатива --csv) |
| `--epochs` | int | `4000` | Количество эпох обучения |
| `--name` | str | `"test"` | Имя эксперимента (папка для сохранения результатов) |
| `--device` | str | `"auto"` | Устройство: `cpu`, `cuda`, `auto` |
| `--batch_size` | int | `256` | Размер батча для обучения |
| `--amp` | flag | `False` | Включить mixed precision (AMP) для CUDA |

### 7.2 Параметры для производных

| Флаг | Тип | По умолчанию | Описание |
|------|-----|--------------|----------|
| `--xi_csv` | str | `None` | CSV с xi1,xi2,xi3,dpsidksi* для включения dψ/dξ в лосс |
| `--dpsi_weight` | float | `0.0` | Вес слагаемого лосса по dψ/dξ |

### 7.3 Якорные слагаемые (физические ограничения)

| Флаг | Тип | По умолчанию | Описание |
|------|-----|--------------|----------|
| `--anchor_SI_weight` | float | `0.0` | Вес якоря для напряжений при C=I: \|\|S(I)\|\|² |
| `--anchor_dpsi0_weight` | float | `0.0` | Вес якоря для производных при ξ=0: \|\|∂ψ/∂ξ(0)\|\|² |
| `--anchor_psi0_weight` | float | `0.0` | Вес нормировки энергии: ψ(0)=0 |

### 7.4 Дополнительные параметры

| Флаг | Тип | По умолчанию | Описание |
|------|-----|--------------|----------|
| `--load_model` | str | `None` | Путь к .pth файлу: только построить графики, без обучения |
| `--visualisation_dpsi` | str | `None` | Путь к CSV файлу для визуализации производных |

---

## 8. Примеры использования

### 8.1 Базовое обучение
```bash
python train_clann_v2.py \
  --csv data/FS_pairs.csv \
  --epochs 2000 \
  --name my_experiment
```

### 8.2 Обучение с якорными слагаемыми
```bash
python train_clann_v2.py \
  --csv data/FS_pairs.csv \
  --epochs 2000 \
  --anchor_SI_weight 1e-2 \
  --anchor_dpsi0_weight 1e-3 \
  --anchor_psi0_weight 1e-3 \
  --name anchor_model
```

### 8.3 Обучение с производными
```bash
python train_clann_v2.py \
  --csv data/FS_pairs.csv \
  --xi_csv data/xi_derivatives.csv \
  --dpsi_weight 0.1 \
  --epochs 2000 \
  --name with_derivatives
```

### 8.4 Обучение на GPU с AMP
```bash
python train_clann_v2.py \
  --csv data/FS_pairs.csv \
  --device cuda \
  --amp \
  --epochs 2000 \
  --name gpu_training
```

### 8.5 Только визуализация (без обучения)
```bash
python train_clann_v2.py \
  --load_model experiments/my_model/my_model.pth \
  --csv data/test_data.csv \
  --name visualization_only
```

---

## 9. Рекомендации по весам якорей

### 9.1 Консервативные (мягкие якоря)
```bash
--anchor_SI_weight 1e-3 \
--anchor_dpsi0_weight 1e-4 \
--anchor_psi0_weight 1e-4
```

### 9.2 Умеренные (баланс)
```bash
--anchor_SI_weight 1e-2 \
--anchor_dpsi0_weight 1e-3 \
--anchor_psi0_weight 1e-3
```

### 9.3 Агрессивные (сильные якоря)
```bash
--anchor_SI_weight 1e-1 \
--anchor_dpsi0_weight 1e-2 \
--anchor_psi0_weight 1e-2
```

**Примечание:** Слишком большие веса могут "перебить" основной лосс на данных. Начните с умеренных значений и корректируйте по результатам.
