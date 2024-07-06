from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel, QScrollArea, QGridLayout, QLineEdit, QPushButton, QFileDialog, QDialog, QFormLayout

class Screen2(QWidget):
    def __init__(self, parent):
        super().__init__(parent)
        self.parent = parent
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()

        self.header = QLabel("Валидация регистраций (редактирование отчёта)")
        layout.addWidget(self.header)

        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_widget = QWidget()
        self.scroll_layout = QVBoxLayout()

        # Mock data for registration cards
        for i in range(5):  # Example with 5 registrations
            registration_card = self.create_registration_card(i)
            self.scroll_layout.addWidget(registration_card)

        self.scroll_widget.setLayout(self.scroll_layout)
        self.scroll_area.setWidget(self.scroll_widget)

        layout.addWidget(self.scroll_area)

        self.report_button = QPushButton("Получить отчет")
        self.report_button.clicked.connect(self.save_report)
        layout.addWidget(self.report_button)

        self.setLayout(layout)

    def create_registration_card(self, index):
        card = QWidget()
        card_layout = QFormLayout()

        trap_number = QLineEdit(f"Ловушка № {index}")
        start_time = QLineEdit(f"Время начала регистрации: {index}")
        end_time = QLineEdit(f"Время конца регистрации: {index}")
        animal_class = QLineEdit(f"Класс (животного) {index}")
        max_count = QLineEdit(f"Количество (животных в кадре максимум): {index}")

        card_layout.addRow("Ловушка №", trap_number)
        card_layout.addRow("Время начала регистрации:", start_time)
        card_layout.addRow("Время конца регистрации:", end_time)
        card_layout.addRow("Класс (животного):", animal_class)
        card_layout.addRow("Количество (животных в кадре максимум):", max_count)

        # Thumbnail gallery
        gallery_layout = QGridLayout()
        for j in range(3):  # Example with 3 thumbnails per registration
            thumbnail = QLabel(f"Image {j}")
            thumbnail.setFixedSize(100, 100)
            gallery_layout.addWidget(thumbnail, j // 3, j % 3)
        card_layout.addRow(gallery_layout)

        card.setLayout(card_layout)
        return card

    def save_report(self):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getSaveFileName(self, "Сохранить отчет", "", "CSV Files (*.csv)", options=options)
        if file_path:
            # Apply changes and save the report
            pass
        self.parent.show_screen(3)
