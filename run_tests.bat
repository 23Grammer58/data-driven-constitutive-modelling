@echo off
echo ========================================
echo Запуск тестов для обученной модели
echo ========================================

set MODEL_PATH=experiments\log_xi\log_xi.pth
set CSV_PATH=dd_tables/maltise_NH_complex/maltise_NH_complex/maltise_NH_1_1/xi_stretch_1_1_01_txt.csv
set DEVICE=cuda

echo.
echo [1/5] Тест монотонности S(C)...
python tests/test_monotonicity.py --pth %MODEL_PATH% --device %DEVICE%
if %errorlevel% neq 0 (
    echo ОШИБКА в тесте монотонности!
    pause
    exit /b 1
)

echo.
echo [2/5] Тест выпуклости ψ(ξ)...
python tests/test_convexity.py --pth %MODEL_PATH% --device %DEVICE%
if %errorlevel% neq 0 (
    echo ОШИБКА в тесте выпуклости!
    pause
    exit /b 1
)

echo.
echo [3/5] Тест корректности градиента DerivWrapper1L...
python tests/test_wrapper_grad.py --pth %MODEL_PATH% --device %DEVICE%
if %errorlevel% neq 0 (
    echo ОШИБКА в тесте градиента!
    pause
    exit /b 1
)

echo.
echo [4/5] Тест корректности HessianWrapper1L...
python tests/test_wrapper_hessian.py --pth %MODEL_PATH% --device %DEVICE%
if %errorlevel% neq 0 (
    echo ОШИБКА в тесте HessianWrapper!
    pause
    exit /b 1
)

echo.
echo [5/5] Тест сравнения с табличными dψ/dξ...
python tests/test_wrapper_on_xi.py --pth %MODEL_PATH% --csv %CSV_PATH% --device %DEVICE% --max_rows 1000
if %errorlevel% neq 0 (
    echo ОШИБКА в тесте сравнения с табличными данными!
    pause
    exit /b 1
)

echo.
echo ========================================
echo ✅ Все тесты успешно завершены!
echo ========================================
pause 