"""
ML-based Adaptive Preprocessing System 
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.model_selection import LeaveOneOut, cross_val_score
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_squared_error, r2_score
from skimage.metrics import structural_similarity as ssim, peak_signal_noise_ratio as psnr
import pandas as pd
import pickle
import warnings
warnings.filterwarnings('ignore')

from analyze import ImageQualityAnalyzer
from handle import DICOMHandler

class SmallDatasetOptimizedML:
    """
    ML system specifically optimized for very small datasets (10-15 images)
    """
    
    def __init__(self):
        self.quality_analyzer = ImageQualityAnalyzer()
        self.dicom_handler = DICOMHandler()
        
        self.parameter_predictors = {
            'brightness_gamma': RidgeCV(alphas=[0.1, 1.0, 10.0, 100.0], cv=3),
            'contrast_clip': RidgeCV(alphas=[0.1, 1.0, 10.0, 100.0], cv=3),
            'noise_strength': RidgeCV(alphas=[0.1, 1.0, 10.0, 100.0], cv=3),
        }
        
        # Robust scaler for small datasets
        self.scaler = RobustScaler()
        self.is_trained = False
        
        print("Small Dataset ML System initialized for 10-15 images")
        print("Focus: Classical ML + extensive synthetic data + robust validation")
    
    def generate_extensive_synthetic_data(self, original_images, n_synthetic_per_image=50):
        """
        Generate extensive synthetic data - more samples per original image
        With 13 images × 50 = 650 synthetic samples
        """
        print(f"Generating {n_synthetic_per_image} synthetic samples per original image...")
        print(f"Total synthetic dataset: {len(original_images)} × {n_synthetic_per_image} = {len(original_images) * n_synthetic_per_image} samples")
        
        synthetic_data = []
        
        # Define systematic degradation parameters
        degradation_configs = [
            {'type': 'noise', 'noise_std': 5}, {'type': 'noise', 'noise_std': 10},
            {'type': 'noise', 'noise_std': 15}, {'type': 'noise', 'noise_std': 20},
            {'type': 'noise', 'noise_std': 25},
            
            # Blur variations
            {'type': 'blur', 'sigma': 0.5}, {'type': 'blur', 'sigma': 1.0},
            {'type': 'blur', 'sigma': 1.5}, {'type': 'blur', 'sigma': 2.0},
            
            # Brightness variations
            {'type': 'brightness', 'gamma': 0.5}, {'type': 'brightness', 'gamma': 0.7},
            {'type': 'brightness', 'gamma': 1.3}, {'type': 'brightness', 'gamma': 1.6},
            {'type': 'brightness', 'gamma': 2.0},
            
            # Contrast variations
            {'type': 'contrast', 'factor': 0.3}, {'type': 'contrast', 'factor': 0.5},
            {'type': 'contrast', 'factor': 0.7}, {'type': 'contrast', 'factor': 1.3},
            
            # Combined degradations
            {'type': 'combined_mild'}, {'type': 'combined_moderate'}, {'type': 'combined_severe'}
        ]
        
        for i, original_image in enumerate(original_images):
            print(f"  Processing original image {i+1}/{len(original_images)}")
            
            for j in range(n_synthetic_per_image):
                # Cycle through degradation configs
                config = degradation_configs[j % len(degradation_configs)]
                
                # Apply degradation
                degraded = self.apply_systematic_degradation(original_image, config)
                
                # Calculate optimal restoration parameters
                optimal_params = self.calculate_optimal_parameters_simple(original_image, degraded)
                
                synthetic_data.append({
                    'degraded_image': degraded,
                    'original_image': original_image,
                    'optimal_params': optimal_params,
                    'degradation_config': config,
                    'source_index': i
                })
        
        print(f"Generated {len(synthetic_data)} synthetic training samples")
        return synthetic_data
    
    def apply_systematic_degradation(self, image, config):
        """Apply systematic degradation based on configuration"""
        degraded = image.copy().astype(np.float32)
        
        if config['type'] == 'noise':
            noise = np.random.normal(0, config['noise_std'], image.shape)
            degraded = np.clip(degraded + noise, 0, 255)
        
        elif config['type'] == 'blur':
            degraded = cv2.GaussianBlur(degraded, (0, 0), config['sigma'])
        
        elif config['type'] == 'brightness':
            normalized = degraded / 255.0
            degraded = np.power(normalized, config['gamma']) * 255.0
        
        elif config['type'] == 'contrast':
            mean_val = np.mean(degraded)
            degraded = (degraded - mean_val) * config['factor'] + mean_val
        
        elif config['type'] == 'combined_mild':
            # Light noise + slight blur
            noise = np.random.normal(0, 8, image.shape)
            degraded = np.clip(degraded + noise, 0, 255)
            degraded = cv2.GaussianBlur(degraded, (3, 3), 0.5)
        
        elif config['type'] == 'combined_moderate':
            # Moderate noise + blur + brightness
            noise = np.random.normal(0, 12, image.shape)
            degraded = np.clip(degraded + noise, 0, 255)
            degraded = cv2.GaussianBlur(degraded, (0, 0), 1.0)
            degraded = np.clip(degraded * 0.8, 0, 255)  # Darken
        
        elif config['type'] == 'combined_severe':
            # Strong degradation
            noise = np.random.normal(0, 18, image.shape)
            degraded = np.clip(degraded + noise, 0, 255)
            degraded = cv2.GaussianBlur(degraded, (0, 0), 1.5)
            normalized = degraded / 255.0
            degraded = np.power(normalized, 1.8) * 255.0  # Strong gamma
        
        return np.clip(degraded, 0, 255).astype(np.uint8)
    
    def calculate_optimal_parameters_simple(self, original, degraded):
        """
        Simplified parameter calculation for faster processing
        """
        # Calculate basic quality differences
        orig_mean = np.mean(original)
        deg_mean = np.mean(degraded)
        
        orig_std = np.std(original)
        deg_std = np.std(degraded)
        
        # Estimate noise level difference
        orig_noise = np.std(cv2.medianBlur(original, 3) - original)
        deg_noise = np.std(cv2.medianBlur(degraded, 3) - degraded)
        
        # Calculate optimal parameters based on differences
        brightness_diff = (orig_mean - deg_mean) / 255.0
        contrast_ratio = orig_std / deg_std if deg_std > 0 else 1.0
        noise_ratio = deg_noise / orig_noise if orig_noise > 0 else 1.0
        
        optimal_params = {
            'brightness_gamma': np.clip(1.0 + brightness_diff * 0.8, 0.3, 3.0),
            'contrast_clip': np.clip(1.0 + (contrast_ratio - 1.0) * 2.0, 1.0, 5.0),
            'noise_strength': np.clip(noise_ratio * 0.8, 0.1, 3.0)
        }
        
        return optimal_params
    
    def extract_essential_features(self, image):
        """
        Extract essential features - reduced set for small datasets
        Focus on most discriminative features to avoid overfitting
        """
        features = []
        
        # Core statistical features (10 features)
        features.extend([
            np.mean(image), np.std(image), np.median(image),
            np.min(image), np.max(image),
            np.percentile(image, 25), np.percentile(image, 75),
            np.var(image), np.ptp(image),  # peak-to-peak
            np.std(image) / np.mean(image) if np.mean(image) > 0 else 0  # CV
        ])
        
        # Histogram features - reduced bins (8 features)
        hist, _ = np.histogram(image, bins=8, range=(0, 256))
        hist_normalized = hist / np.sum(hist)
        features.extend(hist_normalized)
        
        # Edge/gradient features (4 features)
        sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
        gradient_mag = np.sqrt(sobel_x**2 + sobel_y**2)
        
        features.extend([
            np.mean(gradient_mag), np.std(gradient_mag),
            np.sum(gradient_mag > np.percentile(gradient_mag, 90)) / image.size,
            cv2.Laplacian(image, cv2.CV_64F).var()  # Sharpness
        ])
        
        # Noise estimation (2 features)
        median_filtered = cv2.medianBlur(image, 3)
        noise_estimate = np.std(image.astype(np.float32) - median_filtered.astype(np.float32))
        features.extend([
            noise_estimate,
            noise_estimate / np.mean(image) if np.mean(image) > 0 else 0
        ])
        
        # Frequency domain (2 features)
        f_transform = np.fft.fft2(image)
        magnitude = np.abs(f_transform)
        total_energy = np.sum(magnitude)
        
        # High frequency energy
        rows, cols = image.shape
        center_mask = np.zeros((rows, cols), dtype=bool)
        r = min(rows, cols) // 4
        center_mask[rows//2-r:rows//2+r, cols//2-r:cols//2+r] = True
        
        low_freq_energy = np.sum(magnitude[center_mask])
        high_freq_ratio = (total_energy - low_freq_energy) / total_energy if total_energy > 0 else 0
        
        features.extend([
            high_freq_ratio,
            np.log(total_energy + 1)  # Log total energy
        ])
        
        return np.array(features)  # Total: ~26 features
    
    def train_with_leave_one_out(self, images_dir="Images"):
        """
        Train using Leave-One-Out validation - ideal for very small datasets
        """
        print("TRAINING WITH LEAVE-ONE-OUT VALIDATION")
        print("="*50)
        
        # Load original images
        processed_data = self.dicom_handler.process_images_directory(images_dir)
        
        if len(processed_data) < 3:
            print("Error: Need at least 3 images")
            return False
        
        print(f"Loaded {len(processed_data)} original images")
        original_images = [img for _, img, _ in processed_data]
        
        # Generate extensive synthetic data
        synthetic_data = self.generate_extensive_synthetic_data(original_images, n_synthetic_per_image=30)
        
        # Prepare training data
        print("Extracting features...")
        features_list = []
        targets = {param: [] for param in self.parameter_predictors.keys()}
        
        for data in synthetic_data:
            features = self.extract_essential_features(data['degraded_image'])
            features_list.append(features)
            
            params = data['optimal_params']
            for param_name in targets.keys():
                targets[param_name].append(params[param_name])
        
        # Convert to arrays and scale
        X = np.array(features_list)
        X_scaled = self.scaler.fit_transform(X)
        
        print(f"Training data shape: {X_scaled.shape}")
        print("Feature count optimized for small dataset")
        
        # Train models with cross-validation
        self.model_scores = {}
        
        for param_name, model in self.parameter_predictors.items():
            y = np.array(targets[param_name])
            
            # Train model
            model.fit(X_scaled, y)
            
            # Leave-one-out cross-validation on original images
            loo_scores = self.leave_one_out_validation(original_images, param_name)
            
            self.model_scores[param_name] = {
                'loo_r2': np.mean(loo_scores),
                'loo_std': np.std(loo_scores),
                'best_alpha': model.alpha_
            }
            
            print(f"  {param_name}: LOO R² = {np.mean(loo_scores):.3f} ± {np.std(loo_scores):.3f}")
        
        self.is_trained = True
        return True
    
    def leave_one_out_validation(self, original_images, param_name):
        """
        Perform leave-one-out validation on original images
        """
        scores = []
        
        for i in range(len(original_images)):
            # Leave one image out
            train_images = [img for j, img in enumerate(original_images) if j != i]
            test_image = original_images[i]
            
            # Generate synthetic data from training images
            train_synthetic = self.generate_extensive_synthetic_data(train_images, n_synthetic_per_image=20)
            
            # Prepare training data
            X_train = []
            y_train = []
            
            for data in train_synthetic:
                features = self.extract_essential_features(data['degraded_image'])
                X_train.append(features)
                y_train.append(data['optimal_params'][param_name])
            
            X_train = np.array(X_train)
            y_train = np.array(y_train)
            
            # Scale features
            scaler = RobustScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            
            # Train model
            model = RidgeCV(alphas=[0.1, 1.0, 10.0, 100.0], cv=3)
            model.fit(X_train_scaled, y_train)
            
            # Test on held-out image (create degraded version)
            test_degraded = self.apply_systematic_degradation(test_image, {'type': 'combined_moderate'})
            test_features = self.extract_essential_features(test_degraded)
            test_features_scaled = scaler.transform([test_features])
            
            # Predict
            prediction = model.predict(test_features_scaled)[0]
            
            # Calculate ground truth
            ground_truth = self.calculate_optimal_parameters_simple(test_image, test_degraded)[param_name]
            
            # Calculate score (R²-like metric)
            score = 1 - ((prediction - ground_truth) ** 2) / (ground_truth ** 2 + 1e-8)
            scores.append(score)
        
        return scores
    
    def predict_enhancement_parameters(self, image):
        """Predict parameters for new image"""
        if not self.is_trained:
            print("Error: Models not trained")
            return None
        
        features = self.extract_essential_features(image)
        features_scaled = self.scaler.transform([features])
        
        predictions = {}
        for param_name, model in self.parameter_predictors.items():
            predictions[param_name] = model.predict(features_scaled)[0]
        
        return predictions
    
    def enhance_image_ml(self, image):
        """Enhance image using ML predictions"""
        predictions = self.predict_enhancement_parameters(image)
        
        if predictions is None:
            return image, None
        
        enhanced = self.apply_enhancement_pipeline(
            image,
            predictions['brightness_gamma'],
            predictions['contrast_clip'],
            predictions['noise_strength']
        )
        
        return enhanced, predictions
    
    def apply_enhancement_pipeline(self, image, gamma, clip_limit, noise_strength):
        """Apply enhancement with predicted parameters"""
        enhanced = image.copy().astype(np.float32)
        
        # Brightness correction
        if abs(gamma - 1.0) > 0.05:
            normalized = enhanced / 255.0
            enhanced = np.power(normalized, 1.0 / gamma) * 255.0
        
        # Contrast enhancement
        if clip_limit > 1.2:
            enhanced_uint8 = np.clip(enhanced, 0, 255).astype(np.uint8)
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
            enhanced = clahe.apply(enhanced_uint8).astype(np.float32)
        
        # Noise reduction
        if noise_strength > 0.5:
            enhanced_uint8 = np.clip(enhanced, 0, 255).astype(np.uint8)
            if noise_strength > 1.5:
                enhanced = cv2.bilateralFilter(enhanced_uint8, 7, 30, 30).astype(np.float32)
            else:
                enhanced = cv2.GaussianBlur(enhanced_uint8, (3, 3), noise_strength * 0.3).astype(np.float32)
        
        return np.clip(enhanced, 0, 255).astype(np.uint8)
    
    def evaluate_on_small_dataset(self, images_dir="Images"):
        """
        Comprehensive evaluation designed for small datasets
        """
        print("\nSMALL DATASET EVALUATION")
        print("="*40)
        
        processed_data = self.dicom_handler.process_images_directory(images_dir)
        if not processed_data:
            return None
        
        results = []
        
        for filename, original_image, metadata in processed_data:
            print(f"\nEvaluating {filename}...")
            
            # Apply ML enhancement
            enhanced, predictions = self.enhance_image_ml(original_image)
            
            # Calculate metrics
            orig_metrics = self.quality_analyzer.compute_comprehensive_metrics(original_image)
            enh_metrics = self.quality_analyzer.compute_comprehensive_metrics(enhanced)
            
            # Calculate improvement scores
            orig_quality = self.calculate_quality_score(orig_metrics)
            enh_quality = self.calculate_quality_score(enh_metrics)
            improvement = enh_quality - orig_quality
            
            result = {
                'filename': filename,
                'original_quality': orig_quality,
                'enhanced_quality': enh_quality,
                'improvement': improvement,
                'predictions': predictions
            }
            
            results.append(result)
            print(f"  Quality: {orig_quality:.1f} → {enh_quality:.1f} ({improvement:+.1f})")
        
        # Summary statistics
        improvements = [r['improvement'] for r in results]
        print(f"\nSUMMARY:")
        print(f"  Average improvement: {np.mean(improvements):+.2f}")
        print(f"  Std deviation: {np.std(improvements):.2f}")
        print(f"  Images improved: {sum(1 for i in improvements if i > 0)}/{len(improvements)}")
        
        return results
    
    def calculate_quality_score(self, metrics):
        """Calculate quality score"""
        brightness = metrics['brightness']['mean_intensity']
        contrast = metrics['contrast']['rms_contrast']
        sharpness = metrics['sharpness']['laplacian_variance']
        snr = metrics['noise']['estimated_snr_db']
        
        brightness_score = max(0, 100 - abs(brightness - 130) / 130 * 100)
        contrast_score = min(100, contrast / 50 * 100)
        sharpness_score = min(100, sharpness / 500 * 100)
        snr_score = min(100, max(0, snr) / 30 * 100)
        
        return (brightness_score + contrast_score + sharpness_score + snr_score) / 4

def demo_small_dataset_system():
    """Demo optimized for 13 images"""
    print("ML SYSTEM OPTIMIZED FOR 13 IMAGES")
    print("="*40)
    print("Optimizations:")
    print("✓ Ridge regression (more stable than Random Forest)")
    print("✓ Leave-one-out validation")
    print("✓ Extensive synthetic data (30+ per image)")
    print("✓ Reduced feature set (26 features)")
    print("✓ Robust scaling")
    
    system = SmallDatasetOptimizedML()
    success = system.train_with_leave_one_out("Images")
    
    if success:
        print("\n✅ Training successful!")
        results = system.evaluate_on_small_dataset("Images")
        return system
    else:
        print("\n❌ Training failed")
        return None

if __name__ == "__main__":
    demo_small_dataset_system()