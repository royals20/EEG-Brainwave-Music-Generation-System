from PySide6.QtCore import Signal
from PySide6.QtWidgets import QLabel, QPushButton, QVBoxLayout, QWidget


class MusicPanel(QWidget):
    generate_clicked = Signal()

    def __init__(self) -> None:
        super().__init__()
        self.status_label = QLabel("等待生成")
        self.generate_btn = QPushButton("生成脑波音乐")
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("音乐生成"))
        self.generate_btn.clicked.connect(self.generate_clicked.emit)
        layout.addWidget(self.generate_btn)
        layout.addWidget(self.status_label)

    def set_status(self, text: str) -> None:
        self.status_label.setText(text)
