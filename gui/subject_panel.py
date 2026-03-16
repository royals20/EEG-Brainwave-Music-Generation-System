"""
Subject management panel – add / edit / delete / search subjects.
"""

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton,
    QTableWidget, QTableWidgetItem, QHeaderView, QDialog, QFormLayout,
    QComboBox, QSpinBox, QDoubleSpinBox, QMessageBox, QGroupBox,
)

from database.db_manager import DatabaseManager
from utils.file_manager import create_subject_directory, delete_subject_directory
from utils.logger import logger


class SubjectPanel(QWidget):
    """Left-side panel for subject list and management."""

    subject_selected = Signal(str)  # emits subject_id

    def __init__(self, db: DatabaseManager, parent=None) -> None:
        super().__init__(parent)
        self.db = db
        self._build_ui()
        self.refresh()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)

        title = QLabel("Subjects")
        title.setStyleSheet("font-size: 16px; font-weight: bold; color: #1565C0;")
        layout.addWidget(title)

        # Search bar
        search_row = QHBoxLayout()
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Search subjects…")
        self.search_input.textChanged.connect(self.refresh)
        search_row.addWidget(self.search_input)
        layout.addLayout(search_row)

        # Table
        self.table = QTableWidget()
        self.table.setColumnCount(4)
        self.table.setHorizontalHeaderLabels(["ID", "Name", "Age", "Group"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.cellClicked.connect(self._on_row_clicked)
        layout.addWidget(self.table)

        # Buttons
        btn_row = QHBoxLayout()
        for text, slot in [("Add", self._add), ("Edit", self._edit), ("Delete", self._delete)]:
            btn = QPushButton(text)
            btn.setStyleSheet(
                "QPushButton { background: #1976D2; color: white; border-radius: 4px; padding: 6px 14px; }"
                "QPushButton:hover { background: #1565C0; }"
            )
            btn.clicked.connect(slot)
            btn_row.addWidget(btn)
        layout.addLayout(btn_row)

    # ---- Data ----

    def refresh(self) -> None:
        search = self.search_input.text().strip() if hasattr(self, "search_input") else ""
        subjects = self.db.list_subjects(search)
        self.table.setRowCount(len(subjects))
        for row, s in enumerate(subjects):
            self.table.setItem(row, 0, QTableWidgetItem(s.subject_id))
            self.table.setItem(row, 1, QTableWidgetItem(s.name))
            self.table.setItem(row, 2, QTableWidgetItem(str(s.age or "")))
            self.table.setItem(row, 3, QTableWidgetItem(s.group_type or ""))

    def _selected_subject_id(self) -> str | None:
        row = self.table.currentRow()
        if row < 0:
            return None
        item = self.table.item(row, 0)
        return item.text() if item else None

    def _on_row_clicked(self, row: int, _col: int) -> None:
        item = self.table.item(row, 0)
        if item:
            self.subject_selected.emit(item.text())

    # ---- Actions ----

    def _add(self) -> None:
        dlg = _SubjectDialog(self)
        if dlg.exec() == QDialog.Accepted:
            data = dlg.get_data()
            try:
                self.db.add_subject(**data)
                create_subject_directory(data["subject_id"])
                self.refresh()
            except Exception as e:
                QMessageBox.warning(self, "Error", str(e))

    def _edit(self) -> None:
        sid = self._selected_subject_id()
        if sid is None:
            QMessageBox.information(self, "Info", "Select a subject first.")
            return
        subj = self.db.get_subject(sid)
        if subj is None:
            return
        dlg = _SubjectDialog(self, subj)
        if dlg.exec() == QDialog.Accepted:
            data = dlg.get_data()
            data.pop("subject_id", None)
            self.db.update_subject(sid, **data)
            self.refresh()

    def _delete(self) -> None:
        sid = self._selected_subject_id()
        if sid is None:
            QMessageBox.information(self, "Info", "Select a subject first.")
            return
        reply = QMessageBox.question(
            self, "Confirm", f"Delete subject {sid} and all data?",
            QMessageBox.Yes | QMessageBox.No,
        )
        if reply == QMessageBox.Yes:
            self.db.delete_subject(sid)
            delete_subject_directory(sid)
            self.refresh()


class _SubjectDialog(QDialog):
    """Dialog for adding / editing a subject."""

    def __init__(self, parent=None, subject=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Edit Subject" if subject else "Add Subject")
        self.setMinimumWidth(350)
        form = QFormLayout(self)

        self.id_input = QLineEdit(subject.subject_id if subject else "")
        self.id_input.setEnabled(subject is None)
        self.name_input = QLineEdit(subject.name if subject else "")
        self.age_input = QSpinBox()
        self.age_input.setRange(0, 120)
        self.age_input.setValue(subject.age if subject and subject.age else 25)
        self.gender_input = QComboBox()
        self.gender_input.addItems(["Male", "Female", "Other"])
        if subject and subject.gender:
            idx = self.gender_input.findText(subject.gender)
            if idx >= 0:
                self.gender_input.setCurrentIndex(idx)
        self.psqi_input = QDoubleSpinBox()
        self.psqi_input.setRange(0, 21)
        self.psqi_input.setValue(subject.psqi_score if subject and subject.psqi_score else 0)
        self.group_input = QComboBox()
        self.group_input.addItems(["Control", "Experimental", "Other"])
        self.group_input.setEditable(True)
        if subject and subject.group_type:
            self.group_input.setCurrentText(subject.group_type)

        form.addRow("Subject ID:", self.id_input)
        form.addRow("Name:", self.name_input)
        form.addRow("Age:", self.age_input)
        form.addRow("Gender:", self.gender_input)
        form.addRow("PSQI Score:", self.psqi_input)
        form.addRow("Group:", self.group_input)

        btn_row = QHBoxLayout()
        ok = QPushButton("OK")
        ok.clicked.connect(self.accept)
        cancel = QPushButton("Cancel")
        cancel.clicked.connect(self.reject)
        btn_row.addWidget(ok)
        btn_row.addWidget(cancel)
        form.addRow(btn_row)

    def get_data(self) -> dict:
        return {
            "subject_id": self.id_input.text().strip(),
            "name": self.name_input.text().strip(),
            "age": self.age_input.value(),
            "gender": self.gender_input.currentText(),
            "psqi_score": self.psqi_input.value(),
            "group_type": self.group_input.currentText(),
        }
