# **Feature Extraction and Matching - README**  

## **Overview**  
This project involves extracting unique features from a given set of grayscale and color images, generating feature descriptors, and matching features across images using different similarity measures. The goal is to analyze the computational efficiency of these methods.  

## **Tasks to Implement**  

### **1. Feature Extraction using Harris Corner Detector**  
- Apply the **Harris Corner Detector** to extract unique feature points from each image.  
- Report the **computation time** required to generate these feature points.  

### **2. Feature Descriptor Generation using SIFT**  
- Compute **Scale-Invariant Feature Transform (SIFT)** descriptors for the detected feature points.  
- Report the **computation time** required to generate these descriptors.  

### **3. Feature Matching Across Images**  
- Match features between images using:  
  - **Sum of Squared Differences (SSD)**  
  - **Normalized Cross-Correlation (NCC)**  
- Report the **computation time** for feature matching using each method.  

## **Setup & Dependencies**  
Ensure the following Python libraries are installed:  
```bash
pip install numpy opencv-python matplotlib
```
Or, if using Jupyter Notebook, include:  
```python
import numpy as np
import cv2
import matplotlib.pyplot as plt
import time
```

## **How to Run the Code**  
1. Load an input image (grayscale or color).  
2. Run the script for feature extraction and descriptor generation.  
3. Perform feature matching and analyze the results.  
4. Record computation times for each step.  

