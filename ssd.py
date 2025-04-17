import cv2
import numpy as np
import time

class SSDFeatureMatching:
    """Sum of Squared Differences (SSD) feature matching class"""
    
    @staticmethod
    def compute_ssd(descriptor1, descriptor2):
        """
        Compute Sum of Squared Differences between two descriptors.
        Lower value indicates better match.
        
        Args:
            descriptor1: First descriptor vector
            descriptor2: Second descriptor vector
            
        Returns:
            SSD distance value (float)
        """
        # Calculate squared differences
        diff = descriptor1 - descriptor2
        square_diff = np.square(diff)
        
        # Return sum of squared differences
        return np.sum(square_diff)
    
    @staticmethod
    def match_features(descriptors1, descriptors2, threshold=None, k_best=20):
        """
        Match features using SSD distance metric.
        
        Args:
            descriptors1: Array of descriptors from first image
            descriptors2: Array of descriptors from second image
            threshold: Optional threshold for maximum SSD distance
            k_best: Number of best matches to return
            
        Returns:
            List of (index1, index2, distance) tuples sorted by distance
        """
        matches = []
        
        # Calculate SSD distance between each pair of descriptors
        for i, desc1 in enumerate(descriptors1):
            # Find best match for this descriptor in second image
            best_distance = float('inf')
            best_idx = -1
            
            for j, desc2 in enumerate(descriptors2):
                distance = SSDFeatureMatching.compute_ssd(desc1, desc2)
                
                if distance < best_distance:
                    best_distance = distance
                    best_idx = j
            
            # If we found a match below threshold, add it
            if threshold is None or best_distance < threshold:
                matches.append((i, best_idx, best_distance))
        
        # Sort matches by distance (best matches first)
        matches.sort(key=lambda x: x[2])
        
        # Return k best matches
        return matches[:k_best]
    
    @staticmethod
    def apply_ssd_matching(img1, img2):
        """
        Apply SSD feature matching to two images.
        
        Args:
            img1: First image
            img2: Second image
            
        Returns:
            Result image with matches drawn
        """
        # Convert images to grayscale if they aren't already
        if len(img1.shape) == 3:
            gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        else:
            gray1 = img1
            
        if len(img2.shape) == 3:
            gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        else:
            gray2 = img2
        
        # Use SIFT to extract keypoints and descriptors
        sift = cv2.SIFT_create()
        keypoints1, descriptors1 = sift.detectAndCompute(gray1, None)
        keypoints2, descriptors2 = sift.detectAndCompute(gray2, None)
        
        # Match features using SSD
        matches = SSDFeatureMatching.match_features(descriptors1, descriptors2)
        
        # Draw matches
        result_img = SSDFeatureMatching.draw_matches(img1, keypoints1, img2, keypoints2, matches)
        
        return result_img
    
    @staticmethod
    def draw_matches(img1, keypoints1, img2, keypoints2, matches, max_matches=20):
        """
        Draw matches between two images.
        
        Args:
            img1: First image
            keypoints1: Keypoints from first image
            img2: Second image
            keypoints2: Keypoints from second image
            matches: List of (index1, index2, distance) tuples
            max_matches: Maximum number of matches to draw
            
        Returns:
            Image with matches drawn
        """
        # Convert images to BGR if they're grayscale
        if len(img1.shape) == 2:
            img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
        if len(img2.shape) == 2:
            img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
        
        # Get dimensions
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]
        
        # Create output image
        height = max(h1, h2)
        width = w1 + w2
        output = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Place images side by side
        output[:h1, :w1, :] = img1
        output[:h2, w1:w1+w2, :] = img2
        
        # Get a list of random colors for drawing lines
        np.random.seed(42)  # For reproducible results
        colors = np.random.randint(0, 255, (100, 3)).tolist()
        
        # Draw matches
        for idx, (i, j, dist) in enumerate(matches[:max_matches]):
            # Get coordinates
            x1, y1 = map(int, keypoints1[i].pt)
            x2, y2 = map(int, keypoints2[j].pt)
            
            # Draw keypoints
            cv2.circle(output, (x1, y1), 4, (0, 255, 0), -1)
            cv2.circle(output, (x2 + w1, y2), 4, (0, 255, 0), -1)
            
            # Draw line
            color = colors[idx % len(colors)]
            cv2.line(output, (x1, y1), (x2 + w1, y2), color, 1)
        
        return output