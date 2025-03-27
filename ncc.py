import cv2
import numpy as np

class Normal_Cross_Correlation:
    @staticmethod
    def apply_ncc_matching(image, template):
        # Convert the original image to grayscale for matching
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Get dimensions of image and template
        img_h, img_w = gray_image.shape
        temp_h, temp_w = template.shape
        
        # Result matrix to store NCC values
        result = np.zeros((img_h - temp_h + 1, img_w - temp_w + 1))

        # Perform normalized cross-correlation manually
        for y in range(img_h - temp_h + 1):
            for x in range(img_w - temp_w + 1):
                # Extract region of interest (ROI)
                roi = gray_image[y:y+temp_h, x:x+temp_w]
                
                # Compute NCC manually
                mean_roi = np.mean(roi)
                mean_temp = np.mean(template)
                
                numerator = np.sum((roi - mean_roi) * (template - mean_temp))
                denominator = np.sqrt(np.sum((roi - mean_roi) ** 2) * np.sum((template - mean_temp) ** 2))
                
                # Avoid division by zero
                if denominator != 0:
                    result[y, x] = numerator / denominator
                else:
                    result[y, x] = 0

        # Get the location of the best match
        max_val = np.max(result)
        min_val = np.min(result)
        max_loc = np.unravel_index(np.argmax(result), result.shape)
        min_loc = np.unravel_index(np.argmin(result), result.shape)
        
        # Get template size
        h, w = template.shape
        
        # Draw a rectangle around the detected area on the **original colored image**
        cv2.rectangle(image, (max_loc[1], max_loc[0]), (max_loc[1] + w, max_loc[0] + h), (0, 0, 255), 2)

        return image
