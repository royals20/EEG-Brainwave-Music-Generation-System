from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QGroupBox,
                              QPushButton, QTableWidget, QTableWidgetItem,
                              QDialog, QFormLayout, QLineEdit, QComboBox,
                              QSpinBox, QDoubleSpinBox, QDialogButtonBox,
                              QMessageBox, QHeaderView, QLabel)
from PyQt5.QtCore import pyqtSignal, Qt
from datetime import datetime

from database.db_manager import DatabaseManager


class SubjectDialog(QDialog):
    
    def __init__(self, parent=None, subject_data=None):
        super().__init__(parent)
        self.subject_data = subject_data or {}
        self.is_edit = subject_data is not None
        self.init_ui()
    
    def init_ui(self):
        title = '编辑受试者' if self.is_edit else '新增受试者'
        self.setWindowTitle(title)
        self.setMinimumWidth(400)
        
        layout = QVBoxLayout(self)
        form_layout = QFormLayout()
        
        self.subject_id_edit = QLineEdit()
        self.subject_id_edit.setText(self.subject_data.get('subject_id', ''))
        if self.is_edit:
            self.subject_id_edit.setReadOnly(True)
        form_layout.addRow('受试者ID *:', self.subject_id_edit)
        
        self.name_edit = QLineEdit()
        self.name_edit.setText(self.subject_data.get('name', ''))
        form_layout.addRow('姓名 *:', self.name_edit)
        
        self.age_spin = QSpinBox()
        self.age_spin.setRange(0, 150)
        self.age_spin.setValue(self.subject_data.get('age') or 0)
        form_layout.addRow('年龄:', self.age_spin)
        
        self.gender_combo = QComboBox()
        self.gender_combo.addItems(['', '男', '女'])
        gender = self.subject_data.get('gender', '')
        index = self.gender_combo.findText(gender)
        if index >= 0:
            self.gender_combo.setCurrentIndex(index)
        form_layout.addRow('性别:', self.gender_combo)
        
        self.psqi_spin = QDoubleSpinBox()
        self.psqi_spin.setRange(0, 21)
        self.psqi_spin.setDecimals(1)
        self.psqi_spin.setValue(self.subject_data.get('psqi_score') or 0)
        form_layout.addRow('PSQI评分:', self.psqi_spin)
        
        self.group_combo = QComboBox()
        self.group_combo.addItems(['', '实验组', '对照组'])
        group = self.subject_data.get('group_type', '')
        index = self.group_combo.findText(group)
        if index >= 0:
            self.group_combo.setCurrentIndex(index)
        form_layout.addRow('分组:', self.group_combo)
        
        self.record_date_edit = QLineEdit()
        self.record_date_edit.setPlaceholderText('YYYY-MM-DD')
        record_date = self.subject_data.get('record_date', '')
        if record_date:
            self.record_date_edit.setText(record_date)
        else:
            self.record_date_edit.setText(datetime.now().strftime('%Y-%m-%d'))
        form_layout.addRow('记录日期:', self.record_date_edit)
        
        layout.addLayout(form_layout)
        
        buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        )
        buttons.accepted.connect(self.validate_and_accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
    
    def validate_and_accept(self):
        if not self.subject_id_edit.text().strip():
            QMessageBox.warning(self, '验证错误', '请输入受试者ID')
            return
        if not self.name_edit.text().strip():
            QMessageBox.warning(self, '验证错误', '请输入姓名')
            return
        
        self.accept()
    
    def get_data(self):
        return {
            'subject_id': self.subject_id_edit.text().strip(),
            'name': self.name_edit.text().strip(),
            'age': self.age_spin.value() if self.age_spin.value() > 0 else None,
            'gender': self.gender_combo.currentText() or None,
            'psqi_score': self.psqi_spin.value() if self.psqi_spin.value() > 0 else None,
            'group_type': self.group_combo.currentText() or None,
            'record_date': self.record_date_edit.text().strip() or None
        }


class SubjectPanel(QWidget):
    
    subject_selected = pyqtSignal(dict)
    
    def __init__(self, db: DatabaseManager, parent=None):
        super().__init__(parent)
        self.db = db
        self.current_subject = None
        self.init_ui()
        self.load_subjects()
    
    def init_ui(self):
        layout = QVBoxLayout(self)
        
        group_box = QGroupBox('受试者管理')
        group_layout = QVBoxLayout(group_box)
        
        button_layout = QHBoxLayout()
        
        self.add_btn = QPushButton('新增患者')
        self.add_btn.clicked.connect(self.add_subject)
        button_layout.addWidget(self.add_btn)
        
        self.edit_btn = QPushButton('编辑')
        self.edit_btn.clicked.connect(self.edit_subject)
        self.edit_btn.setEnabled(False)
        button_layout.addWidget(self.edit_btn)
        
        self.delete_btn = QPushButton('删除')
        self.delete_btn.clicked.connect(self.delete_subject)
        self.delete_btn.setEnabled(False)
        button_layout.addWidget(self.delete_btn)
        
        group_layout.addLayout(button_layout)
        
        search_layout = QHBoxLayout()
        self.search_edit = QLineEdit()
        self.search_edit.setPlaceholderText('搜索受试者...')
        self.search_edit.textChanged.connect(self.search_subjects)
        search_layout.addWidget(self.search_edit)
        group_layout.addLayout(search_layout)
        
        self.table = QTableWidget()
        self.table.setColumnCount(5)
        self.table.setHorizontalHeaderLabels(['ID', '姓名', '年龄', '性别', '分组'])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setSelectionMode(QTableWidget.SingleSelection)
        self.table.itemSelectionChanged.connect(self.on_selection_changed)
        self.table.doubleClicked.connect(self.select_subject)
        group_layout.addWidget(self.table)
        
        self.select_btn = QPushButton('选择患者')
        self.select_btn.clicked.connect(self.select_subject)
        self.select_btn.setEnabled(False)
        group_layout.addWidget(self.select_btn)
        
        layout.addWidget(group_box)
    
    def load_subjects(self, subjects=None):
        if subjects is None:
            subjects = self.db.get_all_subjects()
        
        self.table.setRowCount(len(subjects))
        
        for row, subject in enumerate(subjects):
            self.table.setItem(row, 0, QTableWidgetItem(subject.get('subject_id', '')))
            self.table.setItem(row, 1, QTableWidgetItem(subject.get('name', '')))
            self.table.setItem(row, 2, QTableWidgetItem(str(subject.get('age', ''))))
            self.table.setItem(row, 3, QTableWidgetItem(subject.get('gender', '')))
            self.table.setItem(row, 4, QTableWidgetItem(subject.get('group_type', '')))
    
    def search_subjects(self, keyword):
        if keyword:
            subjects = self.db.search_subjects(keyword)
        else:
            subjects = self.db.get_all_subjects()
        self.load_subjects(subjects)
    
    def add_subject(self):
        dialog = SubjectDialog(self)
        if dialog.exec_() == QDialog.Accepted:
            data = dialog.get_data()
            if self.db.add_subject(**data):
                self.load_subjects()
                QMessageBox.information(self, '成功', '受试者添加成功')
            else:
                QMessageBox.warning(self, '错误', '受试者ID已存在')
    
    def edit_subject(self):
        row = self.table.currentRow()
        if row < 0:
            return
        
        subject_id = self.table.item(row, 0).text()
        subject = self.db.get_subject(subject_id)
        
        if subject:
            dialog = SubjectDialog(self, subject)
            if dialog.exec_() == QDialog.Accepted:
                data = dialog.get_data()
                if self.db.update_subject(subject_id, **data):
                    self.load_subjects()
                    QMessageBox.information(self, '成功', '受试者更新成功')
    
    def delete_subject(self):
        row = self.table.currentRow()
        if row < 0:
            return
        
        subject_id = self.table.item(row, 0).text()
        
        reply = QMessageBox.question(
            self, '确认删除',
            f'确定要删除受试者 {subject_id} 吗？',
            QMessageBox.Yes | QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            if self.db.delete_subject(subject_id):
                self.load_subjects()
                QMessageBox.information(self, '成功', '受试者删除成功')
    
    def on_selection_changed(self):
        has_selection = self.table.currentRow() >= 0
        self.edit_btn.setEnabled(has_selection)
        self.delete_btn.setEnabled(has_selection)
        self.select_btn.setEnabled(has_selection)
    
    def select_subject(self):
        row = self.table.currentRow()
        if row < 0:
            return
        
        subject_id = self.table.item(row, 0).text()
        self.current_subject = self.db.get_subject(subject_id)
        
        if self.current_subject:
            self.subject_selected.emit(self.current_subject)
    
    def get_current_subject(self):
        return self.current_subject
