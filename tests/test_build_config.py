import os


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _read_text(filename: str) -> str:
    path = os.path.join(PROJECT_ROOT, filename)
    with open(path, 'r', encoding='utf-8') as handle:
        return handle.read()


def test_build_spec_collects_mne_and_runtime_dependencies():
    build_spec = _read_text('build.spec')
    requirements = _read_text('requirements.txt').lower()

    assert 'mne>=1.0.0' in requirements
    assert "'mne'," in build_spec
    assert "for package_name in ('PIL', 'matplotlib', 'pygame', 'mne'):" in build_spec


def test_build_scripts_keep_single_controlled_entrypoint():
    build_bat = _read_text('build.bat').lower()
    build_exe_bat = _read_text('build_exe.bat').lower()

    assert 'del /q *.spec' not in build_bat
    assert 'del /q *.spec' not in build_exe_bat
    assert 'requirements.txt' in build_bat
    assert 'pyinstaller' in build_bat
    assert 'call build.bat' in build_exe_bat


def test_readme_documents_csv_sampling_and_build_steps():
    readme = _read_text('README.md').lower()

    assert 'time' in readme
    assert 'timestamp' in readme
    assert 'requirements-dev.txt' in readme
    assert 'pytest -q' in readme
    assert 'build.bat' in readme
