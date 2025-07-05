#!/usr/bin/env python3
"""
Тестирование фабрики архитектур для создания анизотропной модели.
Параметры: степень полинома 2, все три функции активации, 4 инварианта.
"""

import sys
from pathlib import Path

# Добавляем путь к библиотеке
sys.path.insert(0, str(Path(__file__).parent / "src"))

from CANN_torch.models.architecture_factory import (
    InvNetConfig, StrainEnergyConfig, ModelArchitectureConfig, ArchitectureFactory
)
from CANN_torch.pipeline.CANN_pipeline import TrainingConfig, train_and_evaluate

def test_architecture_factory():
    """Тестирование создания анизотропной модели через фабрику."""
    
    print("🔧 Создание конфигурации анизотропной модели...")
    
    # 1) Конфиг инвариантной сети — все 3 активации, полином 2-й степени
    inv_cfg = InvNetConfig(
        activation_functions=["linear", "exp", "ln"],  # все три функции
        polynomial_degree=2,                          # степень полинома 2
        bias=3.0,                                     # смещение для I1,I2
    )
    
    # 2) Энергия деформации — 4 инварианта: I1,I2 (+смещение 3) и I4,I5 (+смещение 1)
    se_cfg = StrainEnergyConfig(
        invariants_config=[3.0, 3.0],   # изотропная модель (I1,I2)
        # invariants_config=[3.0, 3.0, 1.0, 1.0],   # анизотропная модель (I1,I2,I4,I5)
        inv_net_config=inv_cfg,
    )
    
    # 3) Итоговая архитектура — анизотропная модель (4 инварианта)
    arch_cfg = ModelArchitectureConfig(
        architecture_type="auto",
        strain_energy_cfg=se_cfg,
        setAl=True,
        init_alpha=0.1,
    )
    
    print("✅ Конфигурация создана:")
    print(f"   - Функции активации: {inv_cfg.activation_functions}")
    print(f"   - Степень полинома: {inv_cfg.polynomial_degree}")
    print(f"   - Инварианты: {se_cfg.invariants_config}")
    print(f"   - Архитектура: {arch_cfg.architecture_type}")
    
    # 4) Создаём модель через фабрику
    print("\n🏭 Создание модели через фабрику...")
    model = ArchitectureFactory.create_model_architecture(arch_cfg)
    
    print(f"✅ Модель создана: {type(model).__name__}")
    print(f"   - Количество терминов: {model.terms_count}")
    
    # 5) Проверим структуру модели
    print("\n📊 Структура модели:")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   - Общее количество параметров: {total_params}")
    
    # Проверим Psi-модель
    psi_model = model.Psi_model
    print(f"   - Psi-модель: {type(psi_model).__name__}")
    print(f"   - Количество инвариантных сетей: {psi_model.invariants_count}")
    print(f"   - Терминов на инвариант: {psi_model.terms_count}")
    
    return arch_cfg, model

def test_full_pipeline():
    """Тестирование полного пайплайна с фабрикой."""
    
    print("\n🚀 Тестирование полного пайплайна...")
    
    # Создаём конфигурацию
    arch_cfg, _ = test_architecture_factory()
    
    # Конфигурация обучения
    train_cfg = TrainingConfig(
        experiment_dir="data/GoreTex/2/DIC/3",
        train_protocols=["100_100", "uni"],
        test_protocols=["100_100"],
        epochs=2,  # быстрый тест
        batch_size=8,
        architecture_cfg=arch_cfg,  # используем фабрику
        output_root="results_test",
    )
    
    print("✅ Конфигурация обучения создана")
    print(f"   - Данные: {train_cfg.experiment_dir}")
    print(f"   - Протоколы обучения: {train_cfg.train_protocols}")
    print(f"   - Эпохи: {train_cfg.epochs}")
    
    # Запуск обучения
    try:
        results = train_and_evaluate(train_cfg)
        print(f"✅ Обучение завершено успешно!")
        print(f"   - Результаты сохранены в: {results['run_dir']}")
        return results
    except Exception as e:
        print(f"❌ Ошибка при обучении: {e}")
        raise

if __name__ == "__main__":
    print("🧪 Тестирование фабрики архитектур CANN")
    print("=" * 60)
    
    # Тест 1: Создание модели
    arch_cfg, model = test_architecture_factory()
    
    # Тест 2: Полный пайплайн (если есть данные)
    data_dir = Path("data/GoreTex/2/DIC/3")
    if data_dir.exists():
        results = test_full_pipeline()
        print(f"\n🎉 Все тесты прошли успешно!")
    else:
        print(f"\n⚠️  Директория с данными не найдена: {data_dir}")
        print("   Тест создания модели завершён успешно.") 