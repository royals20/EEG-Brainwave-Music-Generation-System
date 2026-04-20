# EEG 脑波音乐生成系统

PyQt5 桌面应用，用于导入 EEG 数据、检测 SWS 并生成对应的 MIDI/WAV 音乐。

## EEG 导入说明

- 支持 `EDF`、`BDF`、`CSV` 三种输入格式。
- `CSV` 仅支持“列=通道、行=采样点”。
- `CSV` 中若包含 `time` 或 `timestamp` 列，系统会优先尝试据此推断采样率。
- 若 `CSV` 没有可用时间列，或时间列不递增/间隔不稳定，导入时会要求手动输入采样率。
- 时间列仅用于推断采样率，不会被当作 EEG 通道导入。

## 运行与测试

安装运行依赖：

```powershell
python -m pip install -r requirements.txt
```

安装测试依赖并运行测试：

```powershell
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

## 构建桌面版

项目使用单一 `build.spec` 作为 PyInstaller 构建入口，打包版保留 `EDF/BDF` 导入能力。

执行构建：

```powershell
build.bat
```

构建成功后，可执行文件位于：

```text
dist\EEG脑波音乐生成系统.exe
```
