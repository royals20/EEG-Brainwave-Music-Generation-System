@echo off
setlocal
chcp 65001 >nul

echo ========================================
echo   EEG Brainwave Music Generator - Build
echo ========================================
echo.

echo [1/4] Installing runtime dependencies...
python -m pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Failed to install runtime dependencies.
    exit /b 1
)

echo.
echo [2/4] Installing PyInstaller...
python -m pip install pyinstaller
if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Failed to install PyInstaller.
    exit /b 1
)

echo.
echo [3/4] Cleaning previous build output...
if exist build rmdir /s /q build
if exist dist rmdir /s /q dist

echo.
echo [4/4] Running PyInstaller...
python -m PyInstaller --clean --noconfirm build.spec
if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Build failed.
    exit /b 1
)

echo.
echo Build completed.
echo Output executable is under dist\
exit /b 0
