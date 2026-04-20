# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_all

block_cipher = None

datas = []
binaries = []
hiddenimports = [
    'PyQt5',
    'PyQt5.QtCore',
    'PyQt5.QtGui',
    'PyQt5.QtWidgets',
    'numpy',
    'scipy',
    'scipy.signal',
    'scipy.io',
    'scipy.io.wavfile',
    'pandas',
    'matplotlib',
    'matplotlib.backends.backend_qt5agg',
    'midiutil',
    'pretty_midi',
    'pygame',
    'fluidsynth',
    'mne',
]

for package_name in ('PIL', 'matplotlib', 'pygame', 'mne'):
    try:
        package_datas, package_binaries, package_hiddenimports = collect_all(package_name)
        datas.extend(package_datas)
        binaries.extend(package_binaries)
        hiddenimports.extend(package_hiddenimports)
    except Exception:
        pass

a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'torch',
        'tensorflow',
        'keras',
        'cv2',
        'sklearn',
        'IPython',
        'jupyter',
        'notebook',
        'sphinx',
        'pytest',
        'numba',
        'llvmlite',
        'pyarrow',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='EEG脑波音乐生成系统',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
