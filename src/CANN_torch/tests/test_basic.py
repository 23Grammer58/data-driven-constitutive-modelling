import torch
import pytest
from CANN_torch import StrainEnergyCANN, Trainer, ExcelDataset

def test_model_creation():
    """Тест создания модели"""
    model = StrainEnergyCANN()
    assert isinstance(model, torch.nn.Module)

def test_trainer_creation():
    """Тест создания тренера"""
    trainer = Trainer(
        model=StrainEnergyCANN(),
        experiment_name="test"
    )
    assert isinstance(trainer, Trainer)

def test_data_loading():
    """Тест загрузки данных"""
    try:
        dataset = ExcelDataset("test_data.xlsx")
        assert len(dataset) > 0
    except FileNotFoundError:
        pytest.skip("Test data file not found")

def test_model_forward_pass():
    """Тест прямого прохода модели"""
    model = StrainEnergyCANN()
    batch_size = 1
    input_size = 8  # Размер входных данных
    x = torch.randn(batch_size, input_size)
    output = model(x)
    assert output.shape[0] == batch_size
    assert output.shape[1] == 2  # Размер выходных данных 