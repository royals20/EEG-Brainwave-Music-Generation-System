#!/usr/bin/env python3
"""
EEG Sleep Analysis & Brainwave Music Generation System
=======================================================

Entry point – launch the PySide6 desktop application.

Usage:
    python main.py
"""

import sys
import os

# Ensure the project root is on the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QFont

from gui.main_window import MainWindow
from utils.logger import logger


def main() -> None:
    logger.info("Starting EEG Sleep Analysis & Brainwave Music Generation System")

    app = QApplication(sys.argv)
    app.setApplicationName("EEG Sleep Music System")
    app.setFont(QFont("Segoe UI", 10))

    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
