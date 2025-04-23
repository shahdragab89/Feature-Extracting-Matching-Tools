import cv2
import numpy as np

class Normal_Cross_Correlation:
    @staticmethod
    def apply_ncc_matching(image, template):
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        img_h, img_w = gray_image.shape
        temp_h, temp_w = template.shape

        # Precompute template mean and std
        mean_temp = np.mean(template)
        std_temp = np.std(template)
        template_norm = template - mean_temp

        result = np.zeros((img_h - temp_h + 1, img_w - temp_w + 1), dtype=np.float32)

        # Loop through windows 
        for y in range(result.shape[0]):
            row_slice = gray_image[y:y+temp_h, :]
            for x in range(result.shape[1]):
                roi = row_slice[:, x:x+temp_w]
                mean_roi = np.mean(roi)
                std_roi = np.std(roi)

                if std_roi != 0 and std_temp != 0:
                    roi_norm = roi - mean_roi
                    result[y, x] = np.sum(roi_norm * template_norm) / (std_roi * std_temp)
                else:
                    result[y, x] = 0


        # Get match location
        max_loc = np.unravel_index(np.argmax(result), result.shape)
        h, w = template.shape
        cv2.rectangle(image, (max_loc[1], max_loc[0]), (max_loc[1] + w, max_loc[0] + h), (0, 0, 255), 2)

        return image
