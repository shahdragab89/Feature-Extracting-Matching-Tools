import cv2
import numpy as np
import time
import os


class HarrisFeatureExtractor:
    def __init__(self, block_size=3, k_size=3, k=0.04, threshold=0.01):        
        self.block_size = block_size
        self.k_size = k_size
        self.k = k
        self.threshold = threshold

    def run_batch(self, image_dir, output_dir=None, params=None):
        # Set up default parameters
        if params is None:
            params = {
                'block_size': 2,
                'k_size': 3,
                'k': 0.04,
                'threshold': 0.01
            }
            
        # Initialize feature extractor
        self.feature_extractor = HarrisFeatureExtractor(
            block_size=params.get('block_size', 2),
            k_size=params.get('k_size', 3),
            k=params.get('k', 0.04),
            threshold=params.get('threshold', 0.01)
        )
        
        # Set up output directory
        if output_dir is None:
            output_dir = os.path.join(os.path.dirname(image_dir), 
                                     os.path.basename(image_dir) + "_results")
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        # Get list of image files
        valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')
        image_files = [f for f in os.listdir(image_dir) 
                     if os.path.isfile(os.path.join(image_dir, f)) and 
                     f.lower().endswith(valid_extensions)]
                     
        if not image_files:
            print(f"No valid image files found in {image_dir}")
            return {}
            
        # Process each image
        results = {}
        for image_file in image_files:
            print(f"Processing {image_file}...")
            
            # Load image
            image_path = os.path.join(image_dir, image_file)
            image = cv2.imread(image_path)
            
            if image is None:
                print(f"Failed to load {image_file}, skipping.")
                continue
                
            # Determine if image is color or grayscale
            is_color = len(image.shape) == 3 and image.shape[2] == 3
            
            # Start timing
            start_time = time.time()
            
            # Extract features
            corners, eigenvalues = self.feature_extractor.detect_features(image, is_color)
            
            # End timing
            elapsed_time = time.time() - start_time
            
            # Compute metrics
            metrics = self.feature_extractor.compute_lambda_metrics(eigenvalues)
            
            # Create visualization
            vis_image = self.feature_extractor.visualize_features(image, corners)
            
            # Save results
            base_name = os.path.splitext(image_file)[0]
            
            # Save visualized image
            vis_path = os.path.join(output_dir, f"{base_name}_harris.jpg")
            cv2.imwrite(vis_path, vis_image)
            
            # Generate and save lambda distribution plot
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
            
            # Store results
            results[image_file] = {
                'computation_time': elapsed_time,
                'num_corners': len(corners),
                'metrics': metrics
            }
            
            # Write results to text file
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
    
    def compare_images(self, results):
        if not results:
            return {}
            
        # Extract metrics
        computation_times = []
        num_corners = []
        lambda_min_means = []
        lambda_ratio_means = []
        
        for image_name, image_results in results.items():
            computation_times.append(image_results['computation_time'])
            num_corners.append(image_results['num_corners'])
            lambda_min_means.append(image_results['metrics']['lambda_min_mean'])
            lambda_ratio_means.append(image_results['metrics']['lambda_ratio_mean'])
            
        # Compute comparison metrics
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
        # Make sure block_size is odd
        if self.block_size % 2 == 0:
            self.block_size += 1

        # Convert to grayscale if color
        if is_color and len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
            
        # Convert to float32 for processing
        gray = np.float32(gray)
        
        # Compute Harris corner response
        harris_response = cv2.cornerHarris(gray, self.block_size, self.k_size, self.k)
        
        # Get the second moment matrix (M) elements for eigenvalue computation
        # Using Sobel derivatives
        dx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=self.k_size)
        dy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=self.k_size)
        
        # Compute components of the Harris matrix
        Ixx = cv2.GaussianBlur(dx*dx, (self.block_size, self.block_size), 0)
        Ixy = cv2.GaussianBlur(dx*dy, (self.block_size, self.block_size), 0)
        Iyy = cv2.GaussianBlur(dy*dy, (self.block_size, self.block_size), 0)
        
        # Normalize harris_response between 0 and 1
        normalized_response = harris_response / harris_response.max()
        
        # Threshold the Harris response
        corner_mask = normalized_response > self.threshold
        
        # Get corner coordinates
        corners = []
        eigenvalues = []
        
        y_indices, x_indices = np.where(corner_mask)
        
        for y, x in zip(y_indices, x_indices):
            # Create the second moment matrix for this pixel
            M = np.array([[Ixx[y, x], Ixy[y, x]], 
                         [Ixy[y, x], Iyy[y, x]]])
            
            # Calculate eigenvalues
            eig_vals = np.linalg.eigvals(M)
            
            # Only consider valid eigenvalues (sometimes numerical issues cause complex values)
            if np.isreal(eig_vals).all():
                eig_vals = eig_vals.real
                corners.append((x, y))
                eigenvalues.append((eig_vals[0], eig_vals[1]))
        
        return corners, eigenvalues
    
    def visualize_features(self, image, corners, radius=3, color=(0, 255, 0), thickness=2):
        # Create a copy to avoid modifying the original
        if len(image.shape) == 2:  # Grayscale
            vis_img = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:  # Color
            vis_img = image.copy()
            
        # Draw circles at corner locations
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
            if lambda_max > 0:  # Avoid division by zero
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
    # Simple test case
    import matplotlib.pyplot as plt
    
    # Load test image
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