"""data_pipeline.py
Унифицированный модуль подготовки данных, обучения и тестирования моделей
для задачи идентификации конститутивных зависимостей Gore-Tex.

Модуль спроектирован так, чтобы его можно было легко интегрировать в
бекенд-микросервис. Все основные операции обёрнуты в функции с явными
аргументами и явными типами, что упрощает вызов через API.
"""

from __future__ import annotations

import os
import itertools
import copy
import logging
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple, Sequence, Dict, Any, Union, Callable

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import r2_score
import seaborn as sns
import matplotlib.pyplot as plt

# Логирование — удобно при работе в микросервисной архитектуре
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------------------
# Импорт доменных модулей проекта
# --------------------------------------------------------------------------------------
from CANN_torch.core.trainer import Trainer
from CANN_torch.models.CANN_gpt import ModelArchitecture_I5, SingleInvNet4
from CANN_torch.models.architecture_factory import ModelArchitectureConfig, ArchitectureFactory

__all__ = [
    "TrainingConfig",
    "SimpleTensorDataset",
    "load_experiment_dataframe",
    "create_dataloaders",
    "train_and_evaluate",
]

# --------------------------------------------------------------------------------------
# Конфигурация обучения — удобно передавать объектом или dict-ом
# --------------------------------------------------------------------------------------


@dataclass
class TrainingConfig:
    """Все параметры, необходимые для запуска одного цикла train-test."""

    # Директория с CSV-файлами экспериментов DIC
    experiment_dir: Union[str, Path]

    # Какие подтипы эксперимента взять в train / test
    train_protocols: Union[str, Sequence[str]] = "all"
    test_protocols: Union[str, Sequence[str]] = "all"

    # Гиперпараметры
    batch_size: int = 16
    epochs: int = 10
    learning_rate: float = 1e-3
    l2_reg_coeff: float = 1e-2

    # Классы моделей
    model_cls: Callable[..., torch.nn.Module] = ModelArchitecture_I5  # legacy
    single_inv_net_cls: Callable[..., torch.nn.Module] = SingleInvNet4  # legacy

    # Конфиг для гибкой генерации модели (если задан – имеет приоритет над model_cls)
    architecture_cfg: Optional[ModelArchitectureConfig] = None

    # Аппаратные настройки
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Путь до чекпоинта (если нужно дообучать / валидировать существующую модель)
    checkpoint_path: Optional[Union[str, Path]] = None

    # Каталог, куда сохранять результаты (веса, графики, csv-ы)
    output_root: Union[str, Path] = "../results"

    # Случайное зерно для полной воспроизводимости
    seed: int = 42

    # Пользовательская карта весов: {"experiment_type": weight}
    weights_map: Optional[Dict[str, float]] = None

    def __post_init__(self):
        # Нормализуем пути
        self.experiment_dir = Path(self.experiment_dir).expanduser().resolve()
        self.output_root = Path(self.output_root).expanduser().resolve()
        if isinstance(self.checkpoint_path, (str, Path)) and self.checkpoint_path:
            self.checkpoint_path = Path(self.checkpoint_path).expanduser().resolve()

        # Создадим выходную дир.
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = self.output_root / f"run_{timestamp}"
        self.run_dir.mkdir(parents=True, exist_ok=True)

        # Логируем конфиг
        logger.info("Training configuration: %s", asdict(self))

        # Фиксируем зерна
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)


# --------------------------------------------------------------------------------------
# Загрузка и подготовка данных
# --------------------------------------------------------------------------------------

CSV_COLUMNS_RAW = ["# Xlam", "PX", "Ylam", "PY", "experiment_type"]
CSV_COLUMNS_RENAMED = ["lamx", "Px", "lamy", "Py", "experiment_type"]


def _discover_experiment_files(experiment_dir: Path) -> List[Path]:
    """Собираем список всех CSV-файлов в каталоге экспериментов."""
    files = sorted(p for p in experiment_dir.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"Не найдено CSV-файлов в директории {experiment_dir}")
    logger.info("Найдено %d CSV-файлов экспериментов", len(files))
    return files


def _load_single_csv(file_path: Path) -> pd.DataFrame:
    """Чтение одного CSV и выбор нужных колонок."""
    df = pd.read_csv(file_path)
    # Тип эксперимента берём из имени файла (последние 7 символов перед расширением) либо из столбца
    if "experiment_type" not in df.columns:
        exp_type = file_path.stem[-7:]
        df["experiment_type"] = exp_type
    return df[CSV_COLUMNS_RAW].copy()


def load_experiment_dataframe(experiment_dir: Union[str, Path]) -> pd.DataFrame:
    """Грузим все csv, объединяем в один DataFrame и приводим названия колонок."""
    experiment_dir = Path(experiment_dir).expanduser().resolve()
    frames = [_load_single_csv(p) for p in _discover_experiment_files(experiment_dir)]
    df = pd.concat(frames).reset_index(drop=True)
    df.columns = CSV_COLUMNS_RENAMED  # переименовываем
    # Удаляем NaN и конвертируем в float32 для Torch
    df = df.dropna().astype({"lamx": float, "lamy": float, "Px": float, "Py": float})
    logger.info("Собранный датафрейм: %d строк, колонки: %s", len(df), list(df.columns))
    return df


# --------------------------------------------------------------------------------------
# Dataset & DataLoader
# --------------------------------------------------------------------------------------


class SimpleTensorDataset(Dataset):
    """Dataset: (features)->(lamx, lamy, experiment_type), targets -> (Px, Py)."""

    def __init__(self, df: pd.DataFrame, weight_map: Optional[Dict[str, float]] = None):
        df = df.copy()

        # ── добавляем/вычисляем колонку weight ──────────────────────────────
        if "weight" not in df.columns:
            if weight_map is None:
                df["weight"] = 1.0
            else:
                df["weight"] = df["experiment_type"].map(weight_map).fillna(1.0)

        # tensors
        self.features_xy = torch.as_tensor(df[["lamx", "lamy"]].values, dtype=torch.float32)
        self.targets     = torch.as_tensor(df[["Px", "Py"]].values,  dtype=torch.float32)
        self.exp_type    = df["experiment_type"].values  # list[str]
        self.weights     = torch.as_tensor(df["weight"].values, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.features_xy)

    def __getitem__(self, idx: int):
        # Возвращаем features list, где последние элементы exp_type и weight — совместимо с Trainer
        lamx_lamy = self.features_xy[idx]
        lamx = lamx_lamy[0]
        lamy = lamx_lamy[1]
        return [lamx, lamy, self.exp_type[idx], self.weights[idx]], self.targets[idx]


# noinspection PyTypeChecker

def _select_by_protocol(df: pd.DataFrame, protocols: Union[str, Sequence[str]]) -> pd.DataFrame:
    """Фильтрация DataFrame по значениям experiment_type."""
    if protocols == "all":
        return df
    if isinstance(protocols, str):
        protocols = [protocols]
    return df[df["experiment_type"].isin(protocols)].reset_index(drop=True)


def create_dataloaders(
        df_all: pd.DataFrame,
        batch_size: int,
        train_protocols: Union[str, Sequence[str]] = "all",
        test_protocols: Union[str, Sequence[str]] = "all",
        weight_map: Optional[Dict[str, float]] = None,
) -> Tuple[DataLoader, DataLoader, pd.DataFrame, pd.DataFrame]:
    """Возвращает train_loader, test_loader и соответствующие датафреймы."""
    df_train = _select_by_protocol(df_all, train_protocols)
    df_test = _select_by_protocol(df_all, test_protocols)

    logger.info(
        "Train set: %d, Test set: %d", len(df_train), len(df_test)
    )

    train_ds = SimpleTensorDataset(df_train, weight_map)
    test_ds = SimpleTensorDataset(df_test, weight_map)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, df_train, df_test


# --------------------------------------------------------------------------------------
# Визуализация и метрики
# --------------------------------------------------------------------------------------

def _plot_predictions_by_protocol(df: pd.DataFrame, out_dir: Path, prefix: str = "plot") -> pd.DataFrame:
    """Строит графики P11/P22 vs Stress и возвращает DataFrame c R2 значениями."""
    df = df.copy()
    df.columns = [
        "lambda_x",
        "lambda_y",
        "stress_x",
        "stress_y",
        "experiment_type",
        "P11",
        "P22",
    ]

    exp_types = df["experiment_type"].unique()
    r2s: Dict[str, Tuple[float, float]] = {}

    # Один общий график для всех протоколов по lambda_x
    plt.figure(figsize=(12, 6))
    for exp in exp_types:
        sub = df[df["experiment_type"] == exp]
        sns.lineplot(data=sub, x="lambda_x", y="P11", label=f"P11-{exp}")
        sns.scatterplot(data=sub, x="lambda_x", y="stress_x", marker="o", color="red")
        r2_p11 = max(r2_score(sub["stress_x"], sub["P11"]), 0.0)
        r2s.setdefault(exp, [None, None])[0] = r2_p11
    plt.title("P11 vs Stress X (all protocols)")
    plt.savefig(out_dir / f"{prefix}_all_x.png")
    plt.close()

    # Общий график по lambda_y
    plt.figure(figsize=(12, 6))
    for exp in exp_types:
        sub = df[df["experiment_type"] == exp]
        sns.lineplot(data=sub, x="lambda_y", y="P22", label=f"P22-{exp}")
        sns.scatterplot(data=sub, x="lambda_y", y="stress_y", marker="o", color="blue")
        r2_p22 = max(r2_score(sub["stress_y"], sub["P22"]), 0.0)
        r2s.setdefault(exp, [None, None])[1] = r2_p22
    plt.title("P22 vs Stress Y (all protocols)")
    plt.savefig(out_dir / f"{prefix}_all_y.png")
    plt.close()

    # Формируем DataFrame с R2
    r2_df = pd.DataFrame.from_dict(
        {k: {"PX": v[0], "PY": v[1]} for k, v in r2s.items()}, orient="index"
    ).reset_index().rename(columns={"index": "Category"})
    mean_row = {"Category": "Mean", "PX": r2_df["PX"].mean(), "PY": r2_df["PY"].mean()}
    r2_df = pd.concat([r2_df, pd.DataFrame([mean_row])], ignore_index=True)

    r2_df.to_csv(out_dir / "metrics.csv", index=False)
    logger.info("Сохранены метрики и графики в %s", out_dir)
    return r2_df


# --------------------------------------------------------------------------------------
# Основной рабочий процесс: train → predict → визуализация
# --------------------------------------------------------------------------------------

def train_and_evaluate(cfg: TrainingConfig) -> Dict[str, Any]:
    """Полный цикл обучения и оценки. Возвращает нек. статистику."""

    # 1. Данные
    df_all = load_experiment_dataframe(cfg.experiment_dir)
    train_loader, test_loader, df_train, df_test = create_dataloaders(
        df_all,
        batch_size=cfg.batch_size,
        train_protocols=cfg.train_protocols,
        test_protocols=cfg.test_protocols,
        weight_map=cfg.weights_map,
    )

    # 2. Обучение
    trainer = Trainer(
        epochs=cfg.epochs,
        experiment_name=str(cfg.run_dir.name),
        l2_reg_coeff=cfg.l2_reg_coeff,
        learning_rate=cfg.learning_rate,
        model=ArchitectureFactory.create_model_architecture(cfg.architecture_cfg) if cfg.architecture_cfg else cfg.model_cls,
        SingleInvNet=None if cfg.architecture_cfg else cfg.single_inv_net_cls,
        batch_size=cfg.batch_size,
        checkpoint=cfg.checkpoint_path,
        plot_valid=False,
    )
    trained_model = trainer.train(train_loader, None, weighting_data=False)
    trained_model.to(cfg.device).eval()

    # 3. Инференс на тесте
    preds: List[np.ndarray] = []
    for features, _ in test_loader:
        # features теперь список [lamx, lamy, exp_type, weight]
        lamx = features[0].to(cfg.device).requires_grad_(True)
        lamy = features[1].to(cfg.device).requires_grad_(True)
        out = trained_model((lamx, lamy)).detach().cpu().numpy()
        preds.append(out)
    preds_np = np.concatenate(preds, axis=0)  # shape [N,2]

    # 4. Собираем DataFrame для визуализации
    df_res = df_test.copy().reset_index(drop=True)
    df_res["P11"] = preds_np[:, 0]
    df_res["P22"] = preds_np[:, 1]
    metrics_df = _plot_predictions_by_protocol(df_res, cfg.run_dir)

    # 5. Сохраняем веса / сводную инфу
    torch.save(trained_model.state_dict(), cfg.run_dir / "best_model.pth")

    summary = {
        "run_dir": str(cfg.run_dir),
        "metrics": metrics_df,
        "model_constants": getattr(trained_model, "potential_constants", None),
    }
    row_summary = {
        "train_protocols": cfg.train_protocols,
        "test_protocols": cfg.test_protocols,
        "PX_mean": metrics_df.loc[metrics_df.Category == "Mean", "PX"].values[0],
        "PY_mean": metrics_df.loc[metrics_df.Category == "Mean", "PY"].values[0],
    }
    if summary["model_constants"] is not None:
        row_summary.update({f"c{i}": c for i, c in enumerate(summary["model_constants"], 1)})

    summary_df = pd.DataFrame([row_summary])
    summary_df.to_csv(cfg.run_dir / "summary.csv", index=False)

    logger.info("Цикл train+eval завершён. Итоговая средняя R²: %s", summary_df[["PX_mean", "PY_mean"]].values)
    return summary


# --------------------------------------------------------------------------------------
# CLI для запуска как скрипта
# --------------------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser("Data-driven constitutive modelling pipeline")
    parser.add_argument("--data_dir", required=True, help="Путь к директории с CSV-файлами экспериментов")
    parser.add_argument("--train", default="all", help="Список протоколов через запятую или 'all'")
    parser.add_argument("--test", default="all", help="Список протоколов через запятую или 'all'")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--weights", default=None, help="Строка вида '100_100=1.0,uni=2.0' или путь до csv")
    args = parser.parse_args()

    # Разбираем weights
    weights_map = None
    if args.weights:
        weights_map = {k: float(v) for k,v in (pair.split("=") for pair in args.weights.split(","))}

    cfg = TrainingConfig(
        experiment_dir=args.data_dir,
        train_protocols=args.train.split(",") if args.train != "all" else "all",
        test_protocols=args.test.split(",") if args.test != "all" else "all",
        epochs=args.epochs,
        batch_size=args.batch_size,
        weights_map=weights_map,
    )

    train_and_evaluate(cfg) 