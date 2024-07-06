from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QLineEdit, QProgressBar, QFileDialog, QMessageBox

class Screen1(QWidget):
    def __init__(self, parent):
        super().__init__(parent)
        self.parent = parent
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()

        # First row
        first_row_layout = QHBoxLayout()
        self.label = QLabel("Выберите папку с фотографиями:")
        self.search_icon = QLabel()  # Add QIcon here
        self.browse_button = QPushButton("Обзор")
        self.browse_button.clicked.connect(self.browse_folder)

        first_row_layout.addWidget(self.label)
        first_row_layout.addWidget(self.search_icon)
        first_row_layout.addWidget(self.browse_button)

        # Second row
        second_row_layout = QHBoxLayout()
        self.process_button = QPushButton("Обработать")
        self.progress_bar = QProgressBar()
        second_row_layout.addWidget(self.process_button)
        second_row_layout.addWidget(self.progress_bar)

        self.process_button.clicked.connect(self.process_photos)

        # Third row
        self.archive_button = QPushButton("Архив регистраций")
        self.archive_button.clicked.connect(self.go_to_archive)

        layout.addLayout(first_row_layout)
        layout.addLayout(second_row_layout)
        layout.addWidget(self.archive_button)

        self.setLayout(layout)

    def browse_folder(self):
        folder_path = QFileDialog.getExistingDirectory(self, "Выберите папку")
        if folder_path:
            self.label.setText(f"Выбрана папка: {folder_path}")

    def process_photos(self):
        # Mock processing function
        self.progress_bar.setValue(50)
        # Simulate success
        self.show_success_dialog()

    def show_success_dialog(self):

        msg_box = QMessageBox(self)
        msg_box.setWindowTitle("Успех")
        msg_box.setText("Обработка завершена. Перейти к регистрациям?")
        msg_box.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
        result = msg_box.exec()
        if result == QMessageBox.Yes:
            # self.parent().show_screen(2)
            print(self.parent)
            self.parent.show_screen(2)


    def go_to_archive(self):
        self.parent.show_screen(3)