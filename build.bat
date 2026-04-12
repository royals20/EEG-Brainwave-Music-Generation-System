@echo off
echo Installing dependencies...
pip install -r requirements.txt
pip install pyinstaller

echo.
echo Building executable...
pyinstaller --onefile --windowed --name "EEG脑波音乐生成系统" --icon=icon.ico main.py

echo.
echo Build complete! Check the 'dist' folder for the executable.
pause
