from PyQt5 import QtWidgets
from PyQt5.uic import loadUiType
import sys
from worker import Worker

# Load the UI file
ui, _ = loadUiType("Ui3.ui")

class MainApp(QtWidgets.QMainWindow, ui):
    def __init__(self):
        super(MainApp, self).__init__()
        self.setupUi(self)

        self.harrisOption_radioButton.clicked.connect(lambda:self.handle_radio("Harris"))
        self.normalizedCrossSelection_radioButton.clicked.connect(lambda:self.handle_radio("Cross Section"))
        self.sumOfSquareDifference_radioButton.clicked.connect(lambda:self.handle_radio("Sum of Difference"))
        self.applySIFT_radioButton.clicked.connect(lambda:self.handle_radio("SIFT"))

        # Call handle_radio to apply UI changes as initial state
        self.handle_radio("Harris")

        
    def handle_radio(self, option):
        self.option = option
        match self.option:
            case "Harris":
                self.matching_frame.hide()
                self.inputImage2.hide()
                self.harrisOption_frame.show()
            case "Cross Section":
                self.matching_frame.show()
                self.inputImage2.show()
                self.harrisOption_frame.hide()
            case "Sum of Difference":
                self.matching_frame.show()
                self.inputImage2.show()
                self.harrisOption_frame.hide()       
            case "SIFT":
                self.matching_frame.show()
                self.inputImage2.show()
                self.harrisOption_frame.hide()                                           

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = MainApp()
    window.show()
    sys.exit(app.exec_())