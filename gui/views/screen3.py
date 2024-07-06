from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel, QPushButton, QScrollArea, QFormLayout

class Screen3(QWidget):
    def __init__(self, parent):
        super().__init__(parent)
        self.parent = parent
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()

        self.process_new_button = QPushButton("Обработать новые регистрации")
        self.process_new_button.clicked.connect(self.go_to_screen1)
        layout.addWidget(self.process_new_button)

        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_widget = QWidget()
        self.scroll_layout = QFormLayout()

        # Mock data for past reports
        for i in range(5):  # Example with 5 past reports
            report_label = QLabel(f"Дата обработки: {i}, Количество регистраций: {i*10}, Размер отчета: {i*100}KB")
            edit_button = QPushButton("Редактировать отчет")
            edit_button.clicked.connect(lambda checked, idx=i: self.edit_report(idx))
            self.scroll_layout.addRow(report_label, edit_button)

        self.scroll_widget.setLayout(self.scroll_layout)
        self.scroll_area.setWidget(self.scroll_widget)

        layout.addWidget(self.scroll_area)

        self.setLayout(layout)

    def go_to_screen1(self):
        self.parent.show_screen(1)

    def edit_report(self, index):
        # Logic to load and edit the specific report
        self.parent.show_screen(2)
