import sys
import os
import argparse
import time
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QApplication
from PyQt5.uic import loadUiType
from harris_detector_ui import HarrisDetectorUI
from harris_feature_extractor import HarrisFeatureExtractor

# Load the UI file
ui, _ = loadUiType("Ui3.ui")

class HarrisDetectorApp:
    def __init__(self):
        self.ui = None
        self.feature_extractor = None
    
    def run_gui(self):
        if not self.ui:
            self.ui = HarrisDetectorUI()
        self.ui.show()
    
    def close_gui(self):
        if self.ui:
            self.ui.close()
            self.ui = None
    
    def run_batch(self, image_dir, output_dir, params):
        print(f"Processing images in batch mode from {image_dir}...")
        # Implement batch processing logic here
        return {}
    
    def compare_images(self, results):
        print("Comparing processed images...")
        # Implement comparison logic here
        return {}
    
    def generate_report(self, results, comparison, output_dir):
        report_path = os.path.join(output_dir, "report.txt")
        print(f"Generating report at {report_path}...")
        # Implement report generation logic here
        return report_path

class MainApp(QtWidgets.QMainWindow, ui):
    def __init__(self):
        super(MainApp, self).__init__()
        self.setupUi(self)
        self.harrisApp = None

        self.harrisOption_radioButton.clicked.connect(lambda: self.handle_radio("Harris"))
        self.normalizedCrossSelection_radioButton.clicked.connect(lambda: self.handle_radio("Cross Section"))
        self.sumOfSquareDifference_radioButton.clicked.connect(lambda: self.handle_radio("Sum of Difference"))
        self.applySIFT_radioButton.clicked.connect(lambda: self.handle_radio("SIFT"))
        
        # Set default selection to Cross Section
        self.normalizedCrossSelection_radioButton.setChecked(True)
        self.handle_radio("Cross Section")

    def handle_radio(self, option):
        self.option = option
        match self.option:
            case "Harris":
                self.matching_frame.hide()
                self.inputImage2.hide()
                self.harrisOption_frame.show()
                if not self.harrisApp:
                    self.harrisApp = HarrisDetectorApp()
                self.harrisApp.run_gui()
            case _:
                self.matching_frame.show()
                self.inputImage2.show()
                self.harrisOption_frame.hide()
                if self.harrisApp:
                    self.harrisApp.close_gui()

def main():
    parser = argparse.ArgumentParser(description='Harris Feature Extractor')
    parser.add_argument('--batch', action='store_true', help='Run in batch mode')
    parser.add_argument('--image_dir', type=str, help='Directory containing images to process')
    parser.add_argument('--output_dir', type=str, help='Directory to save results')
    parser.add_argument('--block_size', type=int, default=2, help='Block size for Harris detector')
    parser.add_argument('--k_size', type=int, default=3, help='Kernel size for Sobel operator')
    parser.add_argument('--k', type=float, default=0.04, help='Harris detector free parameter')
    parser.add_argument('--threshold', type=float, default=0.01, help='Threshold for detecting corners')
    
    args = parser.parse_args()
    app = HarrisDetectorApp()
    
    if args.batch and args.image_dir:
        params = {
            'block_size': args.block_size,
            'k_size': args.k_size,
            'k': args.k,
            'threshold': args.threshold
        }
        results = app.run_batch(args.image_dir, args.output_dir, params)
        comparison = app.compare_images(results)
        
        if args.output_dir:
            report_path = app.generate_report(results, comparison, args.output_dir)
            print(f"Report generated at: {report_path}")
    else:
        print("Batch mode not enabled. Run the main application GUI instead.")

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = MainApp()
    window.show()
    sys.exit(app.exec_())