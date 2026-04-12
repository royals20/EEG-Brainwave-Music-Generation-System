@echo off
chcp 65001 >nul
echo ========================================
echo   EEG脑波音乐生成系统 - 打包工具
echo ========================================
echo.

echo [1/3] 清理旧的打包文件...
if exist build rmdir /s /q build
if exist dist rmdir /s /q dist
if exist *.spec del /q *.spec

echo [2/3] 开始打包...
pyinstaller --onefile --windowed --name "EEG脑波音乐生成系统" --hidden-import=mne --hidden-import=pretty_midi --hidden-import=midiutil --hidden-import=pygame --hidden-import=scipy --hidden-import=numpy --hidden-import=pandas --hidden-import=matplotlib --collect-data mne --collect-data matplotlib main.py

if %errorlevel% neq 0 (
    echo.
    echo [错误] 打包失败！
    pause
    exit /b 1
)

echo.
echo [3/3] 打包完成！
echo.
echo 可执行文件位置: dist\EEG脑波音乐生成系统.exe
echo.
pause
