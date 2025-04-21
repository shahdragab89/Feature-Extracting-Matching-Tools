import cv2
import numpy as np
import time
import os


class HarrisFeatureExtractor:
    def __init__(self,  k_size=3, k=0.04, threshold=0.01):        
        self.k_size = k_size
        self.k = k
        self.threshold = threshold

    def run_batch(self, image_dir, output_dir=None, params=None):
        #setting up default parameters
        if params is None:
            params = {
                'k_size': 3, #sobel_filter kernel size
                'k': 0.04,
                'threshold': 0.01
            }
            
        #creating new insstance of HarrisFeatureExtractor with default parameters
        self.feature_extractor = HarrisFeatureExtractor(
            k_size=params.get('k_size', 3),
            k=params.get('k', 0.04),
            threshold=params.get('threshold', 0.01)
        )
        
        #setingt up output directory for resultss
        if output_dir is None:
            output_dir = os.path.join(os.path.dirname(image_dir), 
                                     os.path.basename(image_dir) + "_results")
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        #get a list of image files
        valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')
        image_files = [f for f in os.listdir(image_dir) 
                     if os.path.isfile(os.path.join(image_dir, f)) and 
                     f.lower().endswith(valid_extensions)]
                     
        if not image_files:
            print(f"No valid image files found in {image_dir}")
            return {}
            
        #process each image on its own 
        results = {}
        for image_file in image_files:
            print(f"Processing {image_file}...")
            
            #load image
            image_path = os.path.join(image_dir, image_file)
            image = cv2.imread(image_path)
            
            if image is None:
                print(f"Failed to load {image_file}, skipping.")
                continue
                
            #determine whether color or grayscale
            is_color = len(image.shape) == 3 and image.shape[2] == 3
            
            #calc time taken to extract features
            start_time = time.time()
            #feature_extraction:
            corners, eigenvalues = self.feature_extractor.detect_features(image, is_color)
            #end
            elapsed_time = time.time() - start_time
            
            #compute metricss using eigens
            metrics = self.feature_extractor.compute_lambda_metrics(eigenvalues)
            
            #visualization creation of det4ected corners on oriignal img:
            vis_image = self.feature_extractor.visualize_features(image, corners)
            
            #saving results
            base_name = os.path.splitext(image_file)[0]
            
            #saving visualizations
            vis_path = os.path.join(output_dir, f"{base_name}_harris.jpg")
            cv2.imwrite(vis_path, vis_image)
            
            #generate lambda plots/histograms
            plt.figure(figsize=(10, 8))
            
            plt.subplot(2, 1, 1)
            plt.hist(metrics['lambda_min'], bins=50)
            plt.title('λ- Distribution')
            plt.xlabel('λ- Value')
            plt.ylabel('Frequency')
            
            plt.subplot(2, 1, 2)
            plt.hist(metrics['lambda_ratio'], bins=50)
            plt.title('λ-ratio Distribution (λ-/λ+)')
            plt.xlabel('λ-ratio Value')
            plt.ylabel('Frequency')
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"{base_name}_lambda_dist.png"))
            plt.close()
            
            #store results in results dictionary 
            results[image_file] = {
                'computation_time': elapsed_time,
                'num_corners': len(corners),
                'metrics': metrics
            }
            
            #writes results as text into txt file
            with open(os.path.join(output_dir, f"{base_name}_results.txt"), 'w') as f:
                f.write(f"Results for {image_file}\n")
                f.write(f"Computation Time: {elapsed_time:.4f} seconds\n")
                f.write(f"Number of Harris corners detected: {len(corners)}\n")
                f.write(f"Average λ- value: {metrics['lambda_min_mean']:.4f}\n")
                f.write(f"Standard deviation of λ-: {metrics['lambda_min_std']:.4f}\n")
                f.write(f"Average λ-ratio: {metrics['lambda_ratio_mean']:.4f}\n")
                f.write(f"Standard deviation of λ-ratio: {metrics['lambda_ratio_std']:.4f}\n")
            
            print(f"Processed {image_file}: {len(corners)} corners detected in {elapsed_time:.4f} seconds")
            
        return results
    
    #comparing results accross multiple imgs
    def compare_images(self, results):
        if not results:
            return {}
            
        #extracting metrics
        computation_times = []
        num_corners = []
        lambda_min_means = []
        lambda_ratio_means = []
        
        for image_name, image_results in results.items():
            computation_times.append(image_results['computation_time'])
            num_corners.append(image_results['num_corners'])
            lambda_min_means.append(image_results['metrics']['lambda_min_mean'])
            lambda_ratio_means.append(image_results['metrics']['lambda_ratio_mean'])
            
        #computing comparison metrics using mean and std for compering imgs 
        comparison = {
            'avg_computation_time': np.mean(computation_times),
            'std_computation_time': np.std(computation_times),
            'avg_corners': np.mean(num_corners),
            'std_corners': np.std(num_corners),
            'avg_lambda_min': np.mean(lambda_min_means),
            'std_lambda_min': np.std(lambda_min_means),
            'avg_lambda_ratio': np.mean(lambda_ratio_means),
            'std_lambda_ratio': np.std(lambda_ratio_means)
        }
        
        return comparison
        
    def generate_report(self, results, comparison, output_dir):
        report_path = os.path.join(output_dir, "harris_analysis_report.txt")
        
        with open(report_path, 'w') as f:
            f.write("Harris Feature Extractor Analysis Report\n")
            f.write("=======================================\n\n")
            
            f.write("Summary Statistics\n")
            f.write("-----------------\n")
            f.write(f"Number of images analyzed: {len(results)}\n")
            f.write(f"Average computation time: {comparison['avg_computation_time']:.4f} seconds\n")
            f.write(f"Standard deviation of computation time: {comparison['std_computation_time']:.4f} seconds\n")
            f.write(f"Average number of corners detected: {comparison['avg_corners']:.2f}\n")
            f.write(f"Standard deviation of corners detected: {comparison['std_corners']:.2f}\n")
            f.write(f"Average λ- value across all images: {comparison['avg_lambda_min']:.4f}\n")
            f.write(f"Average λ-ratio across all images: {comparison['avg_lambda_ratio']:.4f}\n\n")
            
            f.write("Individual Image Results\n")
            f.write("----------------------\n")
            
            for image_name, image_results in results.items():
                f.write(f"\nImage: {image_name}\n")
                f.write(f"  Computation time: {image_results['computation_time']:.4f} seconds\n")
                f.write(f"  Number of corners detected: {image_results['num_corners']}\n")
                f.write(f"  Average λ- value: {image_results['metrics']['lambda_min_mean']:.4f}\n")
                f.write(f"  Average λ-ratio: {image_results['metrics']['lambda_ratio_mean']:.4f}\n")
                
        return report_path        
        
    def detect_features(self, image, is_color=True):
        #convert colored img to grayscale 
        if is_color and len(image.shape) == 3:
            gray = 0.299 * image[:,:,0] + 0.587 * image[:,:,1] + 0.114 * image[:,:,2]
        else:
            gray = image.copy()
        gray = np.float32(gray)
        
        #get dims of the img
        height, width = gray.shape
        
        #sobel filter applying for gradients calc
        if self.k_size == 3:
            sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
            sobel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=np.float32)
        elif self.k_size == 5:
            sobel_x = np.array([[-1, -2, 0, 2, 1], 
                            [-4, -8, 0, 8, 4], 
                            [-6, -12, 0, 12, 6], 
                            [-4, -8, 0, 8, 4], 
                            [-1, -2, 0, 2, 1]], dtype=np.float32)
            sobel_y = sobel_x.T  # transpose sobel x for getting sobel y
        else:
            #default to the sobel 3 if higher
            sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
            sobel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=np.float32)

        #apply sobel filters using CONV:
        dx = np.zeros_like(gray)
        dy = np.zeros_like(gray)

        #pad img for convolution
        pad = self.k_size // 2
        padded = np.pad(gray, pad, mode='reflect') # reflect mode in padding adds border around img by reflecting values across the edge
        
        #convolution calculationf for derivatives dx and dy:
        for i in range(height):
            for j in range(width):
                window = padded[i:i+self.k_size, j:j+self.k_size]
                dx[i, j] = np.sum(window * sobel_x)
                dy[i, j] = np.sum(window * sobel_y)
        
        #getting Ixx, Ixy, Iyy
        Ixx = dx * dx
        Ixy = dx * dy
        Iyy = dy * dy
        
        #3x3 gaussian smoothing:
        gaussian_kernel = np.array([ #sigma = 1 by default
            [0.075, 0.124, 0.075],
            [0.124, 0.204, 0.124],
            [0.075, 0.124, 0.075]
        ], dtype=np.float32)  

        #pad Ixx, Ixy, Iyy for smoothing
        pad = 1 #for 3*3
        Ixx_padded = np.pad(Ixx, pad, mode='reflect')
        Ixy_padded = np.pad(Ixy, pad, mode='reflect')
        Iyy_padded = np.pad(Iyy, pad, mode='reflect')

        #apply gaussian smoothing
        Ixx_smooth = np.zeros_like(Ixx)
        Ixy_smooth = np.zeros_like(Ixy)
        Iyy_smooth = np.zeros_like(Iyy)

        for i in range(height):
            for j in range(width):
                #extract 3*3 windows
                window_xx = Ixx_padded[i:i+3, j:j+3]
                window_xy = Ixy_padded[i:i+3, j:j+3]
                window_yy = Iyy_padded[i:i+3, j:j+3]
                
                #apply 3*3 gaussian kernel
                Ixx_smooth[i, j] = np.sum(window_xx * gaussian_kernel)
                Ixy_smooth[i, j] = np.sum(window_xy * gaussian_kernel)
                Iyy_smooth[i, j] = np.sum(window_yy * gaussian_kernel)
        
        #Harris response: R = det(M) - k * trace(M)^2
        #wher M = [Ixx_smooth, Ixy_smooth; Ixy_smooth, Iyy_smooth]
        det_M = Ixx_smooth * Iyy_smooth - Ixy_smooth * Ixy_smooth
        trace_M = Ixx_smooth + Iyy_smooth
        harris_response = det_M - self.k * (trace_M ** 2)
        
        #normalize harris response from 0 to 1
        normalized_response = harris_response / np.max(harris_response) if np.max(harris_response) > 0 else harris_response
        
        #threshold harris response
        corner_mask = normalized_response > self.threshold ##THE USAGE OF SET THRESHOLD
        # creates a binary mask where only pixels with response values above the threshold are considered corners
        
        #get corner coordinates
        corners = []
        eigenvalues = []
        
        y_indices, x_indices = np.where(corner_mask)
        
        for y, x in zip(y_indices, x_indices):
            # Create the second moment matrix for this pixel
            M = np.array([[Ixx_smooth[y, x], Ixy_smooth[y, x]], 
                        [Ixy_smooth[y, x], Iyy_smooth[y, x]]])
            
            #eigenvals calc:
            #for 2*2 matrix M [[a, b], [c, d]], eigenvalues are:
            # λ = (a + d ± sqrt((a-d)^2 + 4bc))/2
            a, b = M[0, 0], M[0, 1]
            c, d = M[1, 0], M[1, 1]
            
            trace = a + d
            determinant = a * d - b * c
            
            # Calculate discriminant
            discriminant = trace**2 - 4 * determinant
            
            # Check if discriminant is non-negative
            if discriminant >= 0:
                # Calculate eigenvalues
                sqrt_discriminant = np.sqrt(discriminant)
                lambda1 = (trace + sqrt_discriminant) / 2
                lambda2 = (trace - sqrt_discriminant) / 2
                
                # Store corner and eigenvalues
                corners.append((x, y))
                eigenvalues.append((lambda1, lambda2))
        
        return corners, eigenvalues
    
    def visualize_features(self, image, corners, radius=3, color=(0, 255, 0), thickness=2):
        #create copy of original img to to avoid modifying it
        if len(image.shape) == 2:  # grayscale case
            vis_img = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:  # color case
            vis_img = image.copy()
            
        #draw green cirvles at corners to detect their locations
        for x, y in corners:
            cv2.circle(vis_img, (x, y), radius, color, thickness)
            
        return vis_img
    
    def compute_lambda_metrics(self, eigenvalues):
        lambda_min_values = []
        lambda_ratio_values = []
        
        for eig_vals in eigenvalues:
            lambda_min = min(eig_vals)
            lambda_max = max(eig_vals)
            
            lambda_min_values.append(lambda_min)
            if lambda_max > 0: 
                lambda_ratio_values.append(lambda_min / lambda_max)
                
        metrics = {
            'lambda_min': lambda_min_values,
            'lambda_min_mean': np.mean(lambda_min_values) if lambda_min_values else 0,
            'lambda_min_std': np.std(lambda_min_values) if lambda_min_values else 0,
            'lambda_ratio': lambda_ratio_values,
            'lambda_ratio_mean': np.mean(lambda_ratio_values) if lambda_ratio_values else 0,
            'lambda_ratio_std': np.std(lambda_ratio_values) if lambda_ratio_values else 0
        }
        
        return metrics
    
    def benchmark(self, image, is_color=True, num_runs=5):
        times = []
        num_corners = []
        
        for _ in range(num_runs):
            start_time = time.time()
            corners, eigenvalues = self.detect_features(image, is_color)
            end_time = time.time()
            
            times.append(end_time - start_time)
            num_corners.append(len(corners))
            
        results = {
            'avg_time': np.mean(times),
            'std_time': np.std(times),
            'min_time': min(times),
            'max_time': max(times),
            'avg_corners': np.mean(num_corners),
            'std_corners': np.std(num_corners)
        }
        
        return results


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    test_image = cv2.imread("test_image.jpg")
    if test_image is None:
        # Create a synthetic test image if no image is available
        test_image = np.zeros((300, 300, 3), dtype=np.uint8)
        # Add some geometric shapes
        cv2.rectangle(test_image, (50, 50), (100, 100), (255, 0, 0), -1)
        cv2.circle(test_image, (200, 150), 30, (0, 255, 0), -1)
        cv2.line(test_image, (100, 200), (200, 250), (0, 0, 255), 5)
    
    # Initialize the feature extractor
    extractor = HarrisFeatureExtractor()
    
    # Detect features
    corners, eigenvalues = extractor.detect_features(test_image)
    
    # Visualize
    vis_img = extractor.visualize_features(test_image, corners)
    
    # Display
    plt.figure(figsize=(10, 8))
    plt.imshow(cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB))
    plt.title(f"Harris Corners: {len(corners)} detected")
    plt.axis('off')
    plt.show()
    
    # Compute and print metrics
    metrics = extractor.compute_lambda_metrics(eigenvalues)
    print(f"Average λ-: {metrics['lambda_min_mean']:.4f}")
    print(f"Average λ-ratio: {metrics['lambda_ratio_mean']:.4f}")
    
    # Benchmark
    bench_results = extractor.benchmark(test_image)
    print(f"Average computation time: {bench_results['avg_time']:.4f} seconds")