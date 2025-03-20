from PyQt5 import QtWidgets
from PyQt5.uic import loadUiType
import sys

# Load the UI file
ui, _ = loadUiType("Ui3.ui")

class MainApp(QtWidgets.QMainWindow, ui):
    def __init__(self):
        super(MainApp, self).__init__()
        self.setupUi(self)

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = MainApp()
    window.show()
    sys.exit(app.exec_())