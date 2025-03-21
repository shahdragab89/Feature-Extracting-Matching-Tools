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
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                            QPushButton, QLabel, QFileDialog, QListWidget, QListWidgetItem,
                            QTabWidget, QTextEdit, QProgressBar, QComboBox, QSpinBox, QDoubleSpinBox)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from harris_feature_extractor import HarrisFeatureExtractor
from matplotlib import pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

# Load the UI file
ui, _ = loadUiType("newUI.ui")

class HarrisDetectorApp:
    def __init__(self):
        self.ui = None
        self.feature_extractor = None
    
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

        self.normalizedCrossSelection_radioButton.clicked.connect(lambda: self.handle_radio("Cross Section"))
        self.sumOfSquareDifference_radioButton.clicked.connect(lambda: self.handle_radio("Sum of Difference"))
        self.applySIFT_radioButton.clicked.connect(lambda: self.handle_radio("SIFT"))
        
        # Set default selection to Cross Section
        self.normalizedCrossSelection_radioButton.setChecked(True)
        self.handle_radio("Cross Section")

        self.load_button.clicked.connect(self.load_images)
        self.blockSize_spin.setRange(2, 15)
        self.blockSize_spin.setValue(2)
        self.blockSize_spin.setSingleStep(1)

        self.ksize_spin.setRange(1, 31)
        self.ksize_spin.setValue(3)
        self.ksize_spin.setSingleStep(2)  # Ensure odd values

        self.k_spin.setRange(0.01, 0.5)
        self.k_spin.setValue(0.04)
        self.k_spin.setSingleStep(0.01)

        self.threshold_spin.setRange(0.0, 1.0)
        self.threshold_spin.setValue(0.01)
        self.threshold_spin.setSingleStep(0.01)

        self.run_button.clicked.connect(self.run_feature_extraction)
        self.run_button.setEnabled(False)

        self.file_list.itemClicked.connect(self.display_selected_image)


        self.original_image_label.setAlignment(Qt.AlignCenter)

        # In the MainApp class
                # Tab for Lambda results
        self.lambda_tab = QWidget()
        lambda_layout = QVBoxLayout(self.lambda_tab)  # Ensure 'lambda_tab' is the correct widget
        self.lambda_figure = plt.figure(figsize=(5, 4))
        self.lambda_canvas = FigureCanvas(self.lambda_figure)
        lambda_layout.addWidget(self.lambda_canvas)

    # Class variables
        self.loaded_images = {}  # Dictionary to store loaded images: {path: image}
        self.current_image_path = None
        self.feature_extractor = None
        self.extraction_thread = None

        self.tabs.addTab(self.original_image_tab, "Original Image")
        self.tabs.addTab(self.harris_tab, "Harris Corners")
        self.tabs.addTab(self.lambda_tab, "λ- Analysis")


    def handle_radio(self, option):
        self.option = option
        match self.option:
            case _:
                self.matching_frame.show()
                self.inputImage2.show()

    def load_images(self):
        file_dialog = QFileDialog()
        image_paths, _ = file_dialog.getOpenFileNames(
            self, "Select Images", "", "Image Files (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)")
        
        if not image_paths:
            return
            
        self.loaded_images = {}
        self.file_list.clear()
        
        for path in image_paths:
            filename = os.path.basename(path)
            self.file_list.addItem(filename)
            self.loaded_images[filename] = path
        
        if image_paths:
            self.run_button.setEnabled(True)
            # Select the first image
            self.file_list.setCurrentRow(0)
            self.display_selected_image(self.file_list.currentItem())

    def display_selected_image(self, item):
        if not item:
            return
            
        filename = item.text()
        image_path = self.loaded_images[filename]
        self.current_image_path = image_path
        
        # Load and display the image
        image = cv2.imread(image_path)
        if image is None:
            self.original_image_label.setText(f"Failed to load image: {filename}")
            return
            
        # Convert to RGB for display
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Create QImage and QPixmap
        h, w, c = image_rgb.shape
        q_image = QImage(image_rgb.data, w, h, w * c, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image)
        
        # Scale if too large
        max_size = 800
        if pixmap.width() > max_size or pixmap.height() > max_size:
            pixmap = pixmap.scaled(max_size, max_size, Qt.KeepAspectRatio)
        
        self.original_image_label.setPixmap(pixmap)
        self.tabs.setCurrentIndex(0)  # Switch to original image tab

        
    def run_feature_extraction(self):
        if not self.loaded_images:
            return
            
        # Get parameters
        block_size = self.blockSize_spin.value()
        k_size = self.ksize_spin.value()
        k = self.k_spin.value()
        threshold = self.threshold_spin.value()
        
        # Initialize feature extractor
        from harris_feature_extractor import HarrisFeatureExtractor
        self.feature_extractor = HarrisFeatureExtractor(
            block_size=block_size,
            k_size=k_size,
            k=k,
            threshold=threshold
        )
        
        # Get selected image path
        current_item = self.file_list.currentItem()
        if not current_item:
            return
        
        filename = current_item.text()
        image_path = self.loaded_images[filename]
        
        
        # Create and start worker thread
        self.extraction_thread = FeatureExtractionThread(
            self.feature_extractor,
            image_path
        )
        
        self.extraction_thread.result_ready.connect(self.display_results)
        self.extraction_thread.start()
        
        
    def display_results(self, results):
        # Unpack results
        harris_image, computation_time, lambda_min_values, lambda_ratio_values = results
        
        # Display Harris corners image
        h, w, c = harris_image.shape
        q_image = QImage(harris_image.data, w, h, w * c, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image)
        
        # Scale if too large
        max_size = 800
        if pixmap.width() > max_size or pixmap.height() > max_size:
            pixmap = pixmap.scaled(max_size, max_size, Qt.KeepAspectRatio)
        
        self.harris_image_label.setPixmap(pixmap)
        
        # Display computation time results
        self.results_text.clear()
        self.results_text.append(f"Computation Time: {computation_time:.4f} seconds")
        self.results_text.append(f"Number of Harris corners detected: {len(lambda_min_values)}")
        
        if lambda_min_values:
            self.results_text.append(f"Average λ- value: {np.mean(lambda_min_values):.4f}")
            self.results_text.append(f"Average λ-ratio: {np.mean(lambda_ratio_values):.4f}")
        
        # Plot lambda distribution
        self.lambda_figure.clear()
        if lambda_min_values:
            ax1 = self.lambda_figure.add_subplot(211)
            ax1.hist(lambda_min_values, bins=50)
            ax1.set_title('λ- Distribution')
            ax1.set_xlabel('λ- Value')
            ax1.set_ylabel('Frequency')
            
            ax2 = self.lambda_figure.add_subplot(212)
            ax2.hist(lambda_ratio_values, bins=50)
            ax2.set_title('λ-ratio Distribution (λ-/λ+)')
            ax2.set_xlabel('λ-ratio Value')
            ax2.set_ylabel('Frequency')
            
            self.lambda_figure.tight_layout()
            self.lambda_canvas.draw()
        
        # Switch to Harris corners tab
        self.tabs.setCurrentIndex(1)


class FeatureExtractionThread(QThread):
    result_ready = pyqtSignal(object)
    
    def __init__(self, feature_extractor, image_path):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.image_path = image_path
        
    def run(self):
        # Load image
        image = cv2.imread(self.image_path)
        if image is None:
            return
            
        is_color = len(image.shape) == 3 and image.shape[2] == 3
            
        
        # Start timing
        start_time = time.time()
        
        # Extract features
        corners, eigenvalues = self.feature_extractor.detect_features(image, is_color)
                
        # Get computation time
        computation_time = time.time() - start_time
        
        # Extract eigenvalues for analysis
        lambda_min_values = []
        lambda_ratio_values = []
        
        if eigenvalues:
            for corner_idx in range(len(corners)):
                lambda_min = min(eigenvalues[corner_idx])
                lambda_max = max(eigenvalues[corner_idx])
                
                if lambda_max > 0:  # Avoid division by zero
                    lambda_min_values.append(lambda_min)
                    lambda_ratio_values.append(lambda_min / lambda_max)
                
        # Create visualization image
        vis_image = self.feature_extractor.visualize_features(image, corners)
        
        # Convert to RGB for display if it's grayscale
        if len(vis_image.shape) == 2 or vis_image.shape[2] == 1:
            vis_image = cv2.cvtColor(vis_image, cv2.COLOR_GRAY2RGB)
        elif vis_image.shape[2] == 3:
            vis_image = cv2.cvtColor(vis_image, cv2.COLOR_BGR2RGB)
                    
        # Return results
        results = (vis_image, computation_time, lambda_min_values, lambda_ratio_values)
        self.result_ready.emit(results)



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