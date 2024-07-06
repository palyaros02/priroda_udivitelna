from PySide6.QtWidgets import QApplication, QMainWindow, QStackedWidget
from views.screen1 import Screen1
from views.screen2 import Screen2
from views.screen3 import Screen3
import sys

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Animal Detection App")

        self.central_widget = QStackedWidget()
        self.setCentralWidget(self.central_widget)

        self.screen1 = Screen1(self)
        self.screen2 = Screen2(self)
        self.screen3 = Screen3(self)

        self.central_widget.addWidget(self.screen1)
        self.central_widget.addWidget(self.screen2)
        self.central_widget.addWidget(self.screen3)

        self.show_screen(1)

    def show_screen(self, screen_number):
        self.central_widget.setCurrentIndex(screen_number - 1)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())