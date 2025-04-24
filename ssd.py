import cv2
import numpy as np
import time

class SSDFeatureMatching:
    @staticmethod
    def compute_ssd_region(image_region, template):
        # Calculate squared differences
        diff = image_region.astype(np.float32) - template.astype(np.float32)
        square_diff = np.square(diff)
        
        return np.sum(square_diff)
    
    @staticmethod
    def apply_ssd_matching(image, template):
        # Convert images to grayscale if they aren't already
        if len(image.shape) == 3:
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            color_image = image.copy()
        else:
            gray_image = image
            color_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            
        if len(template.shape) == 3:
            gray_template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        else:
            gray_template = template
        
        image_height, image_width = gray_image.shape
        template_height, template_width = gray_template.shape
        
        if template_height > image_height or template_width > image_width:
            raise ValueError("Template must be smaller than the image")
        
        best_ssd = float('inf')
        best_location = (0, 0)
        
        print(f"Starting template matching with image size: {image_width}x{image_height}, "
              f"template size: {template_width}x{template_height}")
        
        step = max(1, min(template_width, template_height) // 20)
        
        # First pass: coarse search
        for y in range(0, image_height - template_height + 1, step):
            for x in range(0, image_width - template_width + 1, step):
                # Extract region of interest from the image
                roi = gray_image[y:y+template_height, x:x+template_width]
                
                ssd = SSDFeatureMatching.compute_ssd_region(roi, gray_template)
                
                # Update best match
                if ssd < best_ssd:
                    best_ssd = ssd
                    best_location = (x, y)
        
        # Second pass: refine search around best match
        x_start = max(0, best_location[0] - step)
        y_start = max(0, best_location[1] - step)
        x_end = min(image_width - template_width, best_location[0] + step)
        y_end = min(image_height - template_height, best_location[1] + step)
        
        for y in range(y_start, y_end + 1):
            for x in range(x_start, x_end + 1):
                # Extract region of interest from the image
                roi = gray_image[y:y+template_height, x:x+template_width]
                
                ssd = SSDFeatureMatching.compute_ssd_region(roi, gray_template)
                
                # Update best match
                if ssd < best_ssd:
                    best_ssd = ssd
                    best_location = (x, y)
        
        x, y = best_location
        cv2.rectangle(
            color_image, 
            (x, y), 
            (x + template_width, y + template_height), 
            (0, 0, 255),  
            2  
        )
        
        return color_image