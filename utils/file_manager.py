"""
File management utilities for EEG data and subject directories.
"""

import os
import shutil

from utils.logger import logger

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
SUBJECTS_DIR = os.path.join(DATA_DIR, "subjects")


def ensure_directories() -> None:
    """Create base data directories if they do not exist."""
    for d in [DATA_DIR, SUBJECTS_DIR]:
        os.makedirs(d, exist_ok=True)
    logger.debug("Base directories ensured.")


def create_subject_directory(subject_id: str) -> str:
    """Create a directory for a subject and return its path."""
    path = os.path.join(SUBJECTS_DIR, subject_id)
    os.makedirs(path, exist_ok=True)
    logger.info("Created subject directory: %s", path)
    return path


def get_subject_directory(subject_id: str) -> str:
    """Return the path to a subject's directory."""
    return os.path.join(SUBJECTS_DIR, subject_id)


def delete_subject_directory(subject_id: str) -> None:
    """Remove a subject's directory and all contents."""
    path = os.path.join(SUBJECTS_DIR, subject_id)
    if os.path.exists(path):
        shutil.rmtree(path)
        logger.info("Deleted subject directory: %s", path)


def copy_file_to_subject(src_path: str, subject_id: str) -> str:
    """Copy a file into the subject's directory. Returns the destination path."""
    dest_dir = get_subject_directory(subject_id)
    os.makedirs(dest_dir, exist_ok=True)
    filename = os.path.basename(src_path)
    dest_path = os.path.join(dest_dir, filename)
    shutil.copy2(src_path, dest_path)
    logger.info("Copied %s -> %s", src_path, dest_path)
    return dest_path
