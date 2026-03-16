from pathlib import Path
import shutil
from typing import Optional


class FileManager:
    def __init__(self, base_dir: Optional[Path] = None) -> None:
        project_root = Path(__file__).resolve().parents[2]
        self.base_dir = base_dir or (project_root / "data" / "subjects")
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def ensure_subject_dir(self, subject_id: int) -> Path:
        subject_dir = self.base_dir / str(subject_id)
        subject_dir.mkdir(parents=True, exist_ok=True)
        return subject_dir

    def save_eeg_file(self, subject_id: int, source_path: str) -> Path:
        source = Path(source_path)
        subject_dir = self.ensure_subject_dir(subject_id)
        target = subject_dir / source.name
        shutil.copy2(source, target)
        return target

    def save_output_file(self, subject_id: int, source_path: Path) -> Path:
        subject_dir = self.ensure_subject_dir(subject_id)
        target = subject_dir / source_path.name
        shutil.copy2(source_path, target)
        return target
