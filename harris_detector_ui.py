import sys
import time
import os
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                            QPushButton, QLabel, QFileDialog, QListWidget, QListWidgetItem,
                            QTabWidget, QTextEdit, QProgressBar, QComboBox, QSpinBox, QDoubleSpinBox)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt, QThread, pyqtSignal
import numpy as np
import cv2
from matplotlib import pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

class HarrisDetectorUI(QMainWindow):
    def __init__(self):
        super().__init__()
        
        # Main window properties
        self.setWindowTitle("Harris Feature Extractor")
        self.setGeometry(100, 100, 1200, 800)
        
        # Central widget and main layout
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QVBoxLayout(self.central_widget)
        
        # Set styles
        self.setStyleSheet("""
            QWidget {
                background-color: #f0f0f0;  /* Light grey background */
                color: #333;  /* Dark text */
                font-family: Arial, sans-serif;
            }
            QPushButton {
                background-color: #3D61D3;  /* Blue */
                color: white;
                font-weight: bold;
                padding: 6px 12px;
                font-size: 14px;
                border: 2px solid #F2F2F2;
                border-radius: 8px;
            }
            QPushButton:hover {
                background-color: #1F3A93;  /* Darker Blue */
                border: 2px solid #F2F2F2;
                border-radius: 8px;
            }
            QLabel {
                font-weight: bold;
            }
            QTabWidget {
                background-color: #ffffff;  /* White for tabs */
            }
            QTextEdit {
                background-color: #ffffff;  /* White for text area */
                border: 1px solid #ccc;
            }
            QProgressBar {
                background-color: #e0e0e0;  /* Light grey progress bar background */
                color: #333;  /* Dark text */
            }
        """)
        
        # Top controls layout
        top_controls = QHBoxLayout()
        
        # Load images button
        self.load_button = QPushButton("Load Images")
        self.load_button.clicked.connect(self.load_images)
        top_controls.addWidget(self.load_button)
        
        # Type selection
        self.type_label = QLabel("Image Type:")
        top_controls.addWidget(self.type_label)
        self.type_combo = QComboBox()
        self.type_combo.addItems(["Color", "Grayscale", "Both"])
        top_controls.addWidget(self.type_combo)
        
        # Parameters
        self.blockSize_label = QLabel("Block Size:")
        top_controls.addWidget(self.blockSize_label)
        self.blockSize_spin = QSpinBox()
        self.blockSize_spin.setRange(2, 15)
        self.blockSize_spin.setValue(2)
        self.blockSize_spin.setSingleStep(1)
        top_controls.addWidget(self.blockSize_spin)
        
        self.ksize_label = QLabel("Kernel Size:")
        top_controls.addWidget(self.ksize_label)
        self.ksize_spin = QSpinBox()
        self.ksize_spin.setRange(1, 31)
        self.ksize_spin.setValue(3)
        self.ksize_spin.setSingleStep(2)  # Ensure odd values
        top_controls.addWidget(self.ksize_spin)
        
        self.k_label = QLabel("k:")
        top_controls.addWidget(self.k_label)
        self.k_spin = QDoubleSpinBox()
        self.k_spin.setRange(0.01, 0.5)
        self.k_spin.setValue(0.04)
        self.k_spin.setSingleStep(0.01)
        top_controls.addWidget(self.k_spin)
        
        self.threshold_label = QLabel("Threshold:")
        top_controls.addWidget(self.threshold_label)
        self.threshold_spin = QDoubleSpinBox()
        self.threshold_spin.setRange(0.0, 1.0)
        self.threshold_spin.setValue(0.01)
        self.threshold_spin.setSingleStep(0.01)
        top_controls.addWidget(self.threshold_spin)
        
        # Run button
        self.run_button = QPushButton("Extract Features")
        self.run_button.clicked.connect(self.run_feature_extraction)
        self.run_button.setEnabled(False)
        top_controls.addWidget(self.run_button)
        
        self.main_layout.addLayout(top_controls)
        
        # Middle content (tabs, list, image view)
        middle_content = QHBoxLayout()
        
        # Left panel for file list
        left_panel = QVBoxLayout()
        self.file_list_label = QLabel("Loaded Images:")
        left_panel.addWidget(self.file_list_label)
        
        self.file_list = QListWidget()
        self.file_list.itemClicked.connect(self.display_selected_image)
        left_panel.addWidget(self.file_list)
        
        middle_content.addLayout(left_panel, 1)
        
        # Right panel for image display and results
        right_panel = QVBoxLayout()
        
        # Tabs for original image and results
        self.tabs = QTabWidget()
        
        # Tab for original image
        self.original_tab = QWidget()
        original_layout = QVBoxLayout(self.original_tab)
        self.original_image_label = QLabel("No image selected")
        self.original_image_label.setAlignment(Qt.AlignCenter)
        original_layout.addWidget(self.original_image_label)
        self.tabs.addTab(self.original_tab, "Original Image")
        
        # Tab for Harris corners
        self.harris_tab = QWidget()
        harris_layout = QVBoxLayout(self.harris_tab)
        self.harris_image_label = QLabel("No results yet")
        self.harris_image_label.setAlignment(Qt.AlignCenter)
        harris_layout.addWidget(self.harris_image_label)
        self.tabs.addTab(self.harris_tab, "Harris Corners")
        
        # Tab for Lambda results
        self.lambda_tab = QWidget()
        lambda_layout = QVBoxLayout(self.lambda_tab)
        
        # Use matplotlib for lambda visualization
        self.lambda_figure = plt.figure(figsize=(5, 4))
        self.lambda_canvas = FigureCanvas(self.lambda_figure)
        lambda_layout.addWidget(self.lambda_canvas)
        
        self.tabs.addTab(self.lambda_tab, "λ- Analysis")
        
        right_panel.addWidget(self.tabs)
        
        # Results text area
        self.results_label = QLabel("Computation Results:")
        right_panel.addWidget(self.results_label)
        
        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        right_panel.addWidget(self.results_text)
        
        middle_content.addLayout(right_panel, 3)
        
        self.main_layout.addLayout(middle_content)
        
        # Progress bar at bottom
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        self.main_layout.addWidget(self.progress_bar)
        
        # Class variables
        self.loaded_images = {}  # Dictionary to store loaded images: {path: image}
        self.current_image_path = None
        self.feature_extractor = None
        self.extraction_thread = None

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
        
        # Initialize progress bar
        self.progress_bar.setValue(0)
        
        # Create and start worker thread
        self.extraction_thread = FeatureExtractionThread(
            self.feature_extractor,
            image_path,
            self.type_combo.currentText()
        )
        
        self.extraction_thread.progress_update.connect(self.update_progress)
        self.extraction_thread.result_ready.connect(self.display_results)
        self.extraction_thread.start()
        
    def update_progress(self, value):
        self.progress_bar.setValue(value)
        
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
    progress_update = pyqtSignal(int)
    result_ready = pyqtSignal(object)
    
    def __init__(self, feature_extractor, image_path, image_type):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.image_path = image_path
        self.image_type = image_type
        
    def run(self):
        # Load image
        image = cv2.imread(self.image_path)
        if image is None:
            return
            
        # Process based on image type
        if self.image_type == "Grayscale":
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            is_color = False
        elif self.image_type == "Color":
            is_color = True
        else:  # Both - use original format
            is_color = len(image.shape) == 3 and image.shape[2] == 3
            
        self.progress_update.emit(10)
        
        # Start timing
        start_time = time.time()
        
        # Extract features
        corners, eigenvalues = self.feature_extractor.detect_features(image, is_color)
        
        self.progress_update.emit(60)
        
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
        
        self.progress_update.emit(80)
        
        # Create visualization image
        vis_image = self.feature_extractor.visualize_features(image, corners)
        
        # Convert to RGB for display if it's grayscale
        if len(vis_image.shape) == 2 or vis_image.shape[2] == 1:
            vis_image = cv2.cvtColor(vis_image, cv2.COLOR_GRAY2RGB)
        elif vis_image.shape[2] == 3:
            vis_image = cv2.cvtColor(vis_image, cv2.COLOR_BGR2RGB)
            
        self.progress_update.emit(100)
        
        # Return results
        results = (vis_image, computation_time, lambda_min_values, lambda_ratio_values)
        self.result_ready.emit(results)


def main():
    app = QApplication(sys.argv)
    window = HarrisDetectorUI()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()