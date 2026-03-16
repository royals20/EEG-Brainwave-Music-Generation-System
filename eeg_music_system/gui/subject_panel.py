from typing import Optional

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from eeg_music_system.database.db_manager import DBManager


class SubjectPanel(QWidget):
    subject_changed = Signal(int)

    def __init__(self, db: DBManager) -> None:
        super().__init__()
        self.db = db
        self.search_input = QLineEdit()
        self.list_widget = QListWidget()
        self._build_ui()
        self.refresh_subjects()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("受试者管理"))
        self.search_input.setPlaceholderText("搜索受试者...")
        self.search_input.textChanged.connect(self.refresh_subjects)
        layout.addWidget(self.search_input)
        self.list_widget.currentItemChanged.connect(self._on_subject_selected)
        layout.addWidget(self.list_widget)

        buttons = QHBoxLayout()
        add_btn = QPushButton("新增")
        edit_btn = QPushButton("编辑")
        del_btn = QPushButton("删除")
        add_btn.clicked.connect(self.create_subject)
        edit_btn.clicked.connect(self.edit_subject)
        del_btn.clicked.connect(self.delete_subject)
        buttons.addWidget(add_btn)
        buttons.addWidget(edit_btn)
        buttons.addWidget(del_btn)
        layout.addLayout(buttons)

    def refresh_subjects(self) -> None:
        keyword = self.search_input.text().strip()
        subjects = self.db.list_subjects(keyword)
        self.list_widget.clear()
        for subject in subjects:
            item = QListWidgetItem(f"#{subject.id} {subject.name} ({subject.group_type})")
            item.setData(1, subject.id)
            self.list_widget.addItem(item)

    def _subject_form_values(self, title: str, existing: Optional[dict] = None) -> Optional[dict]:
        dialog = QDialog(self)
        dialog.setWindowTitle(title)
        layout = QVBoxLayout(dialog)
        form = QFormLayout()
        name = QLineEdit(existing["name"] if existing else "")
        age = QSpinBox()
        age.setRange(1, 120)
        age.setValue(existing["age"] if existing else 30)
        gender = QLineEdit(existing["gender"] if existing else "Unknown")
        psqi = QLineEdit(str(existing["psqi_score"]) if existing else "0")
        group_type = QLineEdit(existing["group_type"] if existing else "control")
        form.addRow("姓名", name)
        form.addRow("年龄", age)
        form.addRow("性别", gender)
        form.addRow("PSQI", psqi)
        form.addRow("分组", group_type)
        layout.addLayout(form)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        if dialog.exec() != QDialog.Accepted:
            return None
        return {
            "name": name.text().strip(),
            "age": int(age.value()),
            "gender": gender.text().strip(),
            "psqi_score": float(psqi.text().strip() or 0),
            "group_type": group_type.text().strip() or "control",
        }

    def create_subject(self) -> None:
        values = self._subject_form_values("创建受试者")
        if not values:
            return
        self.db.create_subject(**values)
        self.refresh_subjects()

    def edit_subject(self) -> None:
        item = self.list_widget.currentItem()
        if not item:
            return
        subject_id = item.data(1)
        subject = self.db.get_subject(subject_id)
        if not subject:
            return
        values = self._subject_form_values(
            "编辑受试者",
            {
                "name": subject.name,
                "age": subject.age,
                "gender": subject.gender,
                "psqi_score": subject.psqi_score,
                "group_type": subject.group_type,
            },
        )
        if not values:
            return
        self.db.update_subject(subject_id, **values)
        self.refresh_subjects()

    def delete_subject(self) -> None:
        item = self.list_widget.currentItem()
        if not item:
            return
        subject_id = item.data(1)
        reply = QMessageBox.question(self, "确认", "确定删除该受试者？")
        if reply == QMessageBox.Yes:
            self.db.delete_subject(subject_id)
            self.refresh_subjects()

    def _on_subject_selected(self, current, _previous) -> None:
        if current:
            self.subject_changed.emit(current.data(1))
