import numpy as np
import cv2
from scipy import ndimage
from skimage import filters, feature, measure, restoration
from skimage.restoration import estimate_sigma
import matplotlib.pyplot as plt
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

'''
Approach - combine all metrics into a comprehensive 9-panel visualization with quality scoring (0-100 scale) and automated assessment levels ("Excellent" to "Poor")
'''
class ImageQualityAnalyzer:
    """
    Comprehensive image quality analysis for dental X-ray images
    """
    
    def __init__(self):
        self.quality_metrics = {}
    
    def analyze_brightness(self, image):
        """
        Analyze brightness characteristics of the image and return brightness metrics
        """
        brightness_metrics = {}
        
        brightness_metrics['mean_intensity'] = float(np.mean(image))
  
        brightness_metrics['median_intensity'] = float(np.median(image))

        #if mean >> median → bright outliers present; if mean << median → dark outliers present
        
        hist, bins = np.histogram(image.flatten(), bins=256, range=(0, 255))
        
        # weighted mean for determining optimal gamma correction values
        weighted_mean = np.sum(bins[:-1] * hist) / np.sum(hist)
        brightness_metrics['weighted_mean'] = float(weighted_mean)
        
        brightness_metrics['brightness_p10'] = float(np.percentile(image, 10))
        brightness_metrics['brightness_p90'] = float(np.percentile(image, 90))
        
        dark_threshold = 50
        dark_pixels = np.sum(image < dark_threshold)
        brightness_metrics['darkness_ratio'] = float(dark_pixels / image.size)  #High darkness_ratio → needs brightening
        
        # Brightness uniformity (coefficient of variation)
        brightness_metrics['brightness_cv'] = float(np.std(image) / np.mean(image))
        #low->uniform->contrast adjustment  high->non uniform->local adjustments

        return brightness_metrics
    
    def analyze_contrast(self, image):
        """
        Analyze contrast characteristics using multiple methods
        """
        contrast_metrics = {}
        
        # Standard deviation (global contrast)
        contrast_metrics['std_contrast'] = float(np.std(image))
        
        # RMS contrast (root mean square)->less affected by mean brightness
        mean_intensity = np.mean(image)
        rms_contrast = np.sqrt(np.mean((image - mean_intensity) ** 2))
        contrast_metrics['rms_contrast'] = float(rms_contrast)
        
        # Michelson contrast (for periodic patterns)--> tooth structures
        max_intensity = np.max(image)
        min_intensity = np.min(image)
        if max_intensity + min_intensity > 0:
            michelson = (max_intensity - min_intensity) / (max_intensity + min_intensity)
        else:
            michelson = 0
        contrast_metrics['michelson_contrast'] = float(michelson)
        
        # Weber contrast (local contrast measure) -->object background discrimination
        weber_contrast = (max_intensity - min_intensity) / min_intensity if min_intensity > 0 else 0
        contrast_metrics['weber_contrast'] = float(weber_contrast)
        
        # Histogram-based contrast measures
        hist, _ = np.histogram(image.flatten(), bins=256, range=(0, 255))
        
        # Entropy (measure of information content)-->high entropy->rich detail, low->needs contrast enhancement
        hist_normalized = hist / np.sum(hist)
        hist_normalized = hist_normalized[hist_normalized > 0]  
        entropy = -np.sum(hist_normalized * np.log2(hist_normalized))
        contrast_metrics['entropy'] = float(entropy)
        
        # Dynamic range->affected by outliers
        contrast_metrics['dynamic_range'] = float(max_intensity - min_intensity)
        
        # Contrast percentile range (robust measure)->ignores extreme outliers
        p95 = np.percentile(image, 95)
        p5 = np.percentile(image, 5)
        contrast_metrics['percentile_range'] = float(p95 - p5)
        
        # High contrast ratio (percentage of high contrast pixels)
        # Using Sobel edge detection
        sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
        high_contrast_threshold = np.percentile(gradient_magnitude, 90)
        high_contrast_ratio = np.sum(gradient_magnitude > high_contrast_threshold) / image.size
        contrast_metrics['high_contrast_ratio'] = float(high_contrast_ratio)
        
        return contrast_metrics
    
    def analyze_sharpness(self, image):
        """
        Analyze image sharpness using multiple methods
        """
        sharpness_metrics = {}
        
        # Laplacian variance ->sensitive to noise
        # Higher values indicate sharper images
        laplacian = cv2.Laplacian(image, cv2.CV_64F)
        laplacian_var = laplacian.var()
        sharpness_metrics['laplacian_variance'] = float(laplacian_var)
        
        # Tenengrad (gradient-based sharpness)->less sensitive to noise->for edge clarity
        sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
        tenengrad = np.mean(sobel_x**2 + sobel_y**2)
        sharpness_metrics['tenengrad'] = float(tenengrad)
        
        # Modified Laplacian (alternative sharpness measure)->directionally sensitive laplacian
        kernel = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])
        mod_laplacian = cv2.filter2D(image.astype(np.float64), -1, kernel)
        sharpness_metrics['modified_laplacian'] = float(np.mean(np.abs(mod_laplacian)))
        
        # Brenner function (high frequency content)-->fine vertical details
        brenner = 0
        for i in range(image.shape[0] - 2):
            for j in range(image.shape[1]):
                brenner += (image[i+2, j] - image[i, j])**2
        sharpness_metrics['brenner'] = float(brenner / (image.shape[0] * image.shape[1]))
        
        # High frequency ratio (FFT-based)->distinguishing true detail from noise
        f_transform = np.fft.fft2(image)
        f_shift = np.fft.fftshift(f_transform)
        magnitude_spectrum = np.abs(f_shift)
        
        # Create high frequency mask (outer 20% of frequency domain)
        rows, cols = image.shape
        crow, ccol = rows // 2, cols // 2
        mask = np.ones((rows, cols), dtype=bool)
        r = int(0.4 * min(rows, cols))  # 40% radius for low frequencies
        mask[crow-r:crow+r, ccol-r:ccol+r] = False
        
        high_freq_energy = np.sum(magnitude_spectrum[mask])
        total_energy = np.sum(magnitude_spectrum)
        high_freq_ratio = high_freq_energy / total_energy if total_energy > 0 else 0
        sharpness_metrics['high_freq_ratio'] = float(high_freq_ratio)
        
        # Edge density (number of edges per unit area)->more edges->more usable detail 
        edges = cv2.Canny(image.astype(np.uint8), 50, 150)
        edge_density = np.sum(edges > 0) / image.size
        sharpness_metrics['edge_density'] = float(edge_density)
        
        return sharpness_metrics
    
    def analyze_noise(self, image):
        """
        Analyze noise characteristics in the image
        """
        noise_metrics = {}
        
        # Standard deviation in flat regions (homogeneous areas)
        # Using morphological operations to find flat regions == measure actual noise in areas that should be uniform
        kernel = np.ones((5, 5), np.uint8)
        opened = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
        closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)
        
        # Find flat regions where variation is minimal
        variation = np.abs(image.astype(np.float64) - closed.astype(np.float64))
        flat_threshold = np.percentile(variation, 20)  # Bottom 20% variation
        flat_regions = variation < flat_threshold
        
        if np.sum(flat_regions) > 0:
            noise_in_flat = np.std(image[flat_regions])
        else:
            noise_in_flat = 0
        noise_metrics['noise_in_flat_regions'] = float(noise_in_flat)
        
        # Wavelet-based noise estimation using scikit-image
        try:
            sigma = estimate_sigma(image, average_sigmas=True)
            noise_metrics['wavelet_noise_estimate'] = float(sigma)
        except:
            noise_metrics['wavelet_noise_estimate'] = 0.0
        
        # High frequency noise ratio-->detecting electronic noise
        # Apply Gaussian filter and measure difference
        blurred = cv2.GaussianBlur(image, (5, 5), 1.0)
        high_freq_noise = image.astype(np.float64) - blurred.astype(np.float64)
        noise_metrics['high_freq_noise_std'] = float(np.std(high_freq_noise))
        
        # Signal-to-noise ratio estimation --> overall img quality measure
        signal_power = np.mean(image**2)
        noise_power = np.mean(high_freq_noise**2)
        snr = 10 * np.log10(signal_power / noise_power) if noise_power > 0 else 0
        noise_metrics['estimated_snr_db'] = float(snr)
        
        # Coefficient of variation (relative noise measure)--> Same absolute noise level affects dark images more than bright ones-->adjust denoising based on img brightness
        cv = np.std(image) / np.mean(image) if np.mean(image) > 0 else 0
        noise_metrics['coefficient_of_variation'] = float(cv)
        
        # Noise detection using median filter --> detects impulse noise and outlier pixels
        median_filtered = cv2.medianBlur(image, 3)
        noise_residual = np.abs(image.astype(np.float64) - median_filtered.astype(np.float64))
        noise_metrics['median_noise_residual'] = float(np.mean(noise_residual))
        
        # Impulse noise detection (salt and pepper)
        # Count pixels that are significantly different from neighbors
        kernel = np.ones((3, 3)) / 9
        local_mean = cv2.filter2D(image.astype(np.float64), -1, kernel)
        deviation = np.abs(image.astype(np.float64) - local_mean)
        impulse_threshold = 3 * np.std(deviation)
        impulse_pixels = np.sum(deviation > impulse_threshold)
        noise_metrics['impulse_noise_ratio'] = float(impulse_pixels / image.size)
        
        return noise_metrics
    
    def compute_comprehensive_metrics(self, image):
        """
        Compute all quality metrics for an image and organise in a dict by category
        """
        metrics = {}
      
        metrics['basic'] = {
            'width': image.shape[1],
            'height': image.shape[0],
            'total_pixels': image.size,
            'min_value': float(np.min(image)),
            'max_value': float(np.max(image)),
            'mean_value': float(np.mean(image)),
            'std_value': float(np.std(image))
        }
   
        metrics['brightness'] = self.analyze_brightness(image)
        metrics['contrast'] = self.analyze_contrast(image)
        metrics['sharpness'] = self.analyze_sharpness(image)
        metrics['noise'] = self.analyze_noise(image)

        self.quality_metrics = metrics
        
        return metrics
    
    def visualize_quality_analysis(self, image, metrics, title="Image Quality Analysis"):
        """
        Create comprehensive visualization of image quality analysis
        """
        fig = plt.figure(figsize=(16, 12))
        
        # Original image
        plt.subplot(3, 3, 1)
        plt.imshow(image, cmap='gray', vmin=0, vmax=255)
        plt.title(f'Original Image\n{image.shape[1]}x{image.shape[0]}')
        plt.axis('off')
        
        # Histogram
        plt.subplot(3, 3, 2)
        plt.hist(image.flatten(), bins=50, alpha=0.7, color='blue', edgecolor='black')
        plt.title('Pixel Intensity Distribution')
        plt.xlabel('Pixel Intensity')
        plt.ylabel('Frequency')
        plt.axvline(metrics['brightness']['mean_intensity'], color='red', 
                   linestyle='--', label=f"Mean: {metrics['brightness']['mean_intensity']:.1f}")
        plt.legend()
        
        # Edge detection visualization
        plt.subplot(3, 3, 3)
        edges = cv2.Canny(image.astype(np.uint8), 50, 150)
        plt.imshow(edges, cmap='gray')
        plt.title(f'Edge Detection\nDensity: {metrics["sharpness"]["edge_density"]:.4f}')
        plt.axis('off')
        
        # Gradient magnitude
        plt.subplot(3, 3, 4)
        sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
        gradient = np.sqrt(sobel_x**2 + sobel_y**2)
        plt.imshow(gradient, cmap='hot')
        plt.title(f'Gradient Magnitude\nTenengrad: {metrics["sharpness"]["tenengrad"]:.1f}')
        plt.axis('off')
        
        # Frequency domain
        plt.subplot(3, 3, 5)
        f_transform = np.fft.fft2(image)
        f_shift = np.fft.fftshift(f_transform)
        magnitude_spectrum = np.log(np.abs(f_shift) + 1)
        plt.imshow(magnitude_spectrum, cmap='gray')
        plt.title(f'Frequency Domain\nHF Ratio: {metrics["sharpness"]["high_freq_ratio"]:.4f}')
        plt.axis('off')
        
        # Noise visualization
        plt.subplot(3, 3, 6)
        blurred = cv2.GaussianBlur(image, (5, 5), 1.0)
        noise = image.astype(np.float64) - blurred.astype(np.float64)
        plt.imshow(noise, cmap='gray', vmin=-50, vmax=50)
        plt.title(f'High Freq Noise\nSTD: {metrics["noise"]["high_freq_noise_std"]:.2f}')
        plt.axis('off')
        
        # Quality metrics summary
        plt.subplot(3, 3, 7)
        plt.axis('off')
        
        summary_text = f"""
BRIGHTNESS METRICS:
Mean Intensity: {metrics['brightness']['mean_intensity']:.1f}
Darkness Ratio: {metrics['brightness']['darkness_ratio']:.3f}

CONTRAST METRICS:
RMS Contrast: {metrics['contrast']['rms_contrast']:.1f}
Dynamic Range: {metrics['contrast']['dynamic_range']:.0f}
Entropy: {metrics['contrast']['entropy']:.2f}

SHARPNESS METRICS:
Laplacian Var: {metrics['sharpness']['laplacian_variance']:.1f}
Edge Density: {metrics['sharpness']['edge_density']:.4f}

NOISE METRICS:
SNR (dB): {metrics['noise']['estimated_snr_db']:.1f}
Noise in Flat: {metrics['noise']['noise_in_flat_regions']:.2f}
        """
        
        plt.text(0.05, 0.95, summary_text, transform=plt.gca().transAxes,
                verticalalignment='top', fontfamily='monospace', fontsize=9,
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        # Quality scores bar chart
        plt.subplot(3, 3, 8)
        categories = ['Brightness', 'Contrast', 'Sharpness', 'Noise']
        
        # Normalize some key metrics for comparison (0-100 scale)
        brightness_score = min(100, metrics['brightness']['mean_intensity'] / 255 * 100)
        contrast_score = min(100, metrics['contrast']['rms_contrast'] / 50 * 100)
        sharpness_score = min(100, metrics['sharpness']['laplacian_variance'] / 1000 * 100)
        noise_score = max(0, 100 - metrics['noise']['coefficient_of_variation'] * 100)
        
        scores = [brightness_score, contrast_score, sharpness_score, noise_score]
        colors = ['gold', 'skyblue', 'lightgreen', 'salmon']
        
        bars = plt.bar(categories, scores, color=colors, alpha=0.7)
        plt.title('Quality Scores (0-100)')
        plt.ylabel('Score')
        plt.ylim(0, 100)
        
        # Add value labels on bars
        for bar, score in zip(bars, scores):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{score:.1f}', ha='center', va='bottom')
        
        # Metrics comparison radar chart would go here (subplot 9)
        plt.subplot(3, 3, 9)
        plt.axis('off')
        
        # Add overall quality assessment
        overall_score = np.mean(scores)
        quality_level = "Excellent" if overall_score > 80 else \
                       "Good" if overall_score > 60 else \
                       "Fair" if overall_score > 40 else "Poor"
        
        assessment_text = f"""
OVERALL ASSESSMENT:

Quality Score: {overall_score:.1f}/100
Quality Level: {quality_level}

KEY FINDINGS:
• Brightness: {'Optimal' if brightness_score > 60 else 'Needs adjustment'}
• Contrast: {'Good' if contrast_score > 50 else 'Low'}
• Sharpness: {'Sharp' if sharpness_score > 40 else 'Blurry'}
• Noise Level: {'Low' if noise_score > 70 else 'High'}
        """
        
        plt.text(0.05, 0.95, assessment_text, transform=plt.gca().transAxes,
                verticalalignment='top', fontfamily='monospace', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        plt.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()
    
    def create_metrics_dataframe(self, metrics_list, filenames):
        """
        Convert metrics from multiple images to a pandas DataFrame for analysis
        """
        rows = []
        
        for i, (metrics, filename) in enumerate(zip(metrics_list, filenames)):
            row = {'filename': filename}
            
            # Flatten nested metrics dictionary
            for category, category_metrics in metrics.items():
                if isinstance(category_metrics, dict):
                    for metric_name, value in category_metrics.items():
                        row[f"{category}_{metric_name}"] = value
                else:
                    row[category] = category_metrics
            
            rows.append(row)
        
        return pd.DataFrame(rows)
    
    def plot_metrics_distribution(self, df, metrics_to_plot=None):
        """
        Plot distribution of key metrics across the dataset
        """
        if metrics_to_plot is None:
            # Default key metrics for visualization
            metrics_to_plot = [
                'brightness_mean_intensity',
                'contrast_rms_contrast',
                'sharpness_laplacian_variance',
                'noise_estimated_snr_db'
            ]
        
        # Filter metrics that exist in the dataframe
        available_metrics = [m for m in metrics_to_plot if m in df.columns]
        
        if not available_metrics:
            print("No specified metrics found in dataframe")
            return
        
        n_metrics = len(available_metrics)
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()
        
        for i, metric in enumerate(available_metrics[:4]):  # Limit to 4 plots
            if i < len(axes):
                ax = axes[i]
                values = df[metric].dropna()
                
                # Histogram
                ax.hist(values, bins=20, alpha=0.7, edgecolor='black')
                ax.set_title(f'{metric.replace("_", " ").title()}')
                ax.set_xlabel('Value')
                ax.set_ylabel('Frequency')
                
                # Add statistics
                mean_val = values.mean()
                std_val = values.std()
                ax.axvline(mean_val, color='red', linestyle='--', 
                          label=f'Mean: {mean_val:.2f}')
                ax.axvline(mean_val + std_val, color='orange', linestyle=':', 
                          alpha=0.7, label=f'±1 STD')
                ax.axvline(mean_val - std_val, color='orange', linestyle=':', alpha=0.7)
                ax.legend()
                ax.grid(True, alpha=0.3)
        
        # Hide unused subplots
        for i in range(len(available_metrics), len(axes)):
            axes[i].set_visible(False)
        
        plt.suptitle('Distribution of Image Quality Metrics', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()
        
        # Print summary statistics
        print("\nSUMMARY STATISTICS:")
        print("=" * 50)
        for metric in available_metrics:
            values = df[metric].dropna()
            print(f"{metric.replace('_', ' ').title()}:")
            print(f"  Mean: {values.mean():.3f}")
            print(f"  Std:  {values.std():.3f}")
            print(f"  Min:  {values.min():.3f}")
            print(f"  Max:  {values.max():.3f}")
            print()