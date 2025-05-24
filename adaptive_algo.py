"""
Adaptive Preprocessing Pipeline - algo based
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from analyze import ImageQualityAnalyzer
from handle import DICOMHandler

class AdaptivePreprocessor:
    """
    Adaptive preprocessing pipeline that analyzes image quality metrics and 
    dynamically adjusts preprocessing parameters for optimal results.
    """
    
    def __init__(self):
        self.quality_analyzer = ImageQualityAnalyzer()
        self.dicom_handler = DICOMHandler()
        
        # Consolidated configuration
        self.config = {
            'optimal_ranges': {
                'brightness': {'min': 80, 'max': 180, 'target': 130},
                'contrast': {'min': 30, 'max': 80, 'target': 50},
                'sharpness': {'min': 100, 'max': 1000, 'target': 300},
                'snr': {'min': 20, 'max': 35, 'target': 28}
            },
            'thresholds': {
                'brightness': [60, 80, 180, 200],  # very_dark, dark, bright, very_bright
                'contrast': [25, 45, 70],          # low, moderate, high
                'sharpness': [80, 150, 400],       # blurry, moderate_blur, sharp
                'noise_snr': [15, 22, 30],         # very_noisy, noisy, clean
                'edge_density': [0.02, 0.05]       # poor, moderate, good
            },
            'processing_methods': {
                'brightness': {
                    'very_dark': {'method': 'gamma', 'gamma': 0.6},
                    'dark': {'method': 'gamma', 'gamma': 0.8},
                    'bright': {'method': 'gamma', 'gamma': 1.3},
                    'very_bright': {'method': 'gamma', 'gamma': 1.6},
                    'optimal': {'method': 'none'}
                },
                'contrast': {
                    'low': {'method': 'strong_clahe', 'clip_limit': 4.0, 'grid_size': (8, 8)},
                    'moderate': {'method': 'mild_clahe', 'clip_limit': 2.5, 'grid_size': (6, 6)},
                    'good': {'method': 'mild_clahe', 'clip_limit': 1.8, 'grid_size': (4, 4)},
                    'high': {'method': 'none'}
                },
                'noise': {
                    'very_noisy': {'method': 'bilateral', 'sigma_color': 50, 'sigma_space': 50, 'kernel_size': 9},
                    'noisy': {'method': 'bilateral', 'sigma_color': 30, 'sigma_space': 30, 'kernel_size': 5},
                    'moderate': {'method': 'gaussian', 'kernel_size': 3, 'sigma': 0.5},
                    'clean': {'method': 'none'}
                },
                'sharpening': {
                    'very_blurry': {'method': 'unsharp', 'strength': 2.0, 'radius': 2.0, 'threshold': 0},
                    'blurry': {'method': 'unsharp', 'strength': 1.5, 'radius': 1.5, 'threshold': 0},
                    'adequate': {'method': 'unsharp', 'strength': 0.8, 'radius': 1.0, 'threshold': 0},
                    'sharp': {'method': 'none'}
                }
            }
        }
    
    def _categorize_metric(self, value, metric_type):
        """Universal categorization function for all metrics"""
        thresholds = self.config['thresholds'][metric_type]
        
        if metric_type == 'brightness':
            if value < thresholds[0]: return 'very_dark'
            elif value < thresholds[1]: return 'dark'
            elif value > thresholds[3]: return 'very_bright'
            elif value > thresholds[2]: return 'bright'
            else: return 'optimal'
            
        elif metric_type == 'contrast':
            if value < thresholds[0]: return 'low'
            elif value < thresholds[1]: return 'moderate'
            elif value > thresholds[2]: return 'high'
            else: return 'good'
            
        elif metric_type == 'sharpness':
            if value < thresholds[0]: return 'very_blurry'
            elif value < thresholds[1]: return 'blurry'
            elif value > thresholds[2]: return 'sharp'
            else: return 'adequate'
            
        elif metric_type == 'noise_snr':
            if value < thresholds[0]: return 'very_noisy'
            elif value < thresholds[1]: return 'noisy'
            elif value > thresholds[2]: return 'clean'
            else: return 'moderate'
            
        elif metric_type == 'edge_density':
            if value < thresholds[0]: return 'poor'
            elif value < thresholds[1]: return 'moderate'
            else: return 'good'
    
    def analyze_image_characteristics(self, image):
        """Analyze image and determine its characteristics for adaptive processing"""
        print("  Analyzing image characteristics...")
        
        metrics = self.quality_analyzer.compute_comprehensive_metrics(image)
        
        # Extract key values
        brightness = metrics['brightness']['mean_intensity']
        contrast = metrics['contrast']['rms_contrast']
        sharpness = metrics['sharpness']['laplacian_variance']
        snr = metrics['noise']['estimated_snr_db']
        edge_density = metrics['sharpness']['edge_density']
        noise_in_flat = metrics['noise']['noise_in_flat_regions']
        
        # Categorize using unified function
        characteristics = {
            'brightness_level': self._categorize_metric(brightness, 'brightness'),
            'contrast_level': self._categorize_metric(contrast, 'contrast'),
            'sharpness_level': self._categorize_metric(sharpness, 'sharpness'),
            'noise_level': self._categorize_metric(snr, 'noise_snr'),
            'edge_quality': self._categorize_metric(edge_density, 'edge_density'),
            'metrics': metrics
        }
        
        print(f"    Brightness: {brightness:.1f} ({characteristics['brightness_level']})")
        print(f"    Contrast: {contrast:.1f} ({characteristics['contrast_level']})")
        print(f"    Sharpness: {sharpness:.1f} ({characteristics['sharpness_level']})")
        print(f"    Noise: {snr:.1f}dB ({characteristics['noise_level']})")
        
        return characteristics
    
    def determine_processing_strategy(self, characteristics):
        """Determine optimal processing strategy based on image characteristics"""
        print("  Determining processing strategy...")
        
        strategy = {}
        
        # Get base parameters from configuration
        for process_type in ['brightness', 'contrast', 'noise', 'sharpening']:
            level_key = f'{process_type}_level'
            if process_type == 'noise':
                level_key = 'noise_level'
            elif process_type == 'sharpening':
                level_key = 'sharpness_level'
                
            level = characteristics[level_key]
            base_params = self.config['processing_methods'][process_type][level].copy()
            
            # Apply adaptive adjustments
            strategy[process_type] = self._apply_adaptive_adjustments(
                base_params, process_type, characteristics
            )
        
        # Determine processing order
        strategy['processing_order'] = self._determine_processing_order(characteristics)
        
        return strategy
    
    def _apply_adaptive_adjustments(self, base_params, process_type, characteristics):
        """Apply adaptive adjustments to base parameters"""
        params = base_params.copy()
        
        # Special cases and cross-metric adjustments
        if process_type == 'contrast' and characteristics['brightness_level'] in ['very_dark', 'very_bright']:
            if params['method'] == 'mild_clahe':
                params.update({'method': 'strong_clahe', 'clip_limit': 4.0, 'grid_size': (8, 8)})
                params['reason'] = 'Low contrast with poor brightness needs strong CLAHE'
        
        elif process_type == 'noise':
            # Adjust for high-quality images
            if self._calculate_quality_score(characteristics['metrics']) > 65:
                params.update({'method': 'gaussian', 'kernel_size': 3, 'sigma': 0.3})
                params['reason'] = 'High quality image - preserve detail with minimal denoising'
            # Adjust noise reduction for blurry images
            elif characteristics['noise_level'] == 'noisy' and characteristics['sharpness_level'] in ['very_blurry', 'blurry']:
                params.update({'method': 'gaussian', 'kernel_size': 3, 'sigma': 0.8})
                params['reason'] = 'Noisy and blurry - use gentle Gaussian denoising'
        
        elif process_type == 'sharpening' and characteristics['noise_level'] in ['very_noisy', 'noisy']:
            params.update({'method': 'none'})
            params['reason'] = 'Skip sharpening for noisy images to avoid amplifying noise'
        
        # Add reason if not already set
        if 'reason' not in params:
            params['reason'] = f"{process_type.title()} correction based on {characteristics[f'{process_type}_level' if process_type != 'sharpening' else 'sharpness_level']} level"
        
        return params
    
    def _determine_processing_order(self, characteristics):
        """Determine optimal processing order based on image characteristics"""
        # Default order
        order = ['brightness', 'contrast', 'noise', 'sharpening']
        
        # Adjust for very noisy images - do noise reduction early
        if characteristics['noise_level'] in ['very_noisy', 'noisy']:
            order = ['brightness', 'noise', 'contrast', 'sharpening']
        
        return order
    
    def _apply_processing_step(self, image, process_type, params):
        """Universal processing function for all enhancement types"""
        if params['method'] == 'none':
            return image
        
        if process_type == 'brightness':
            if params['method'] == 'gamma':
                normalized = image.astype(np.float32) / 255.0
                corrected = np.power(normalized, 1.0 / params['gamma'])
                return (corrected * 255).astype(np.uint8)
        
        elif process_type == 'contrast':
            if 'clahe' in params['method']:
                clahe = cv2.createCLAHE(
                    clipLimit=params['clip_limit'],
                    tileGridSize=params['grid_size']
                )
                return clahe.apply(image)
            elif params['method'] == 'adaptive_histogram_eq':
                return cv2.equalizeHist(image)
        
        elif process_type == 'noise':
            if params['method'] == 'bilateral':
                return cv2.bilateralFilter(
                    image, params['kernel_size'],
                    params['sigma_color'], params['sigma_space']
                )
            elif params['method'] == 'gaussian':
                return cv2.GaussianBlur(
                    image, (params['kernel_size'], params['kernel_size']), params['sigma']
                )
        
        elif process_type == 'sharpening':
            if params['method'] == 'unsharp':
                blurred = cv2.GaussianBlur(image, (0, 0), params['radius'])
                sharpened = cv2.addWeighted(
                    image, 1.0 + params['strength'],
                    blurred, -params['strength'], 0
                )
                return np.clip(sharpened, 0, 255).astype(np.uint8)
        
        return image
    
    def process_image(self, image, show_steps=False):
        """Apply complete adaptive preprocessing pipeline to an image"""
        print("Starting adaptive preprocessing...")
        
        # Analyze and determine strategy
        characteristics = self.analyze_image_characteristics(image)
        strategy = self.determine_processing_strategy(characteristics)
        
        # Apply processing steps
        current_image = image.copy()
        steps = {'original': image.copy()}
        processing_log = []
        
        print("  Applying adaptive processing steps...")
        
        for step_name in strategy['processing_order']:
            step_params = strategy[step_name]
            
            if step_params['method'] != 'none':
                print(f"    {step_name.title()}: {step_params['reason']}")
                current_image = self._apply_processing_step(current_image, step_name, step_params)
                steps[f'after_{step_name}'] = current_image.copy()
                processing_log.append(f"{step_name}: {step_params['reason']}")
            else:
                print(f"    {step_name.title()}: Skipped - {step_params['reason']}")
                processing_log.append(f"{step_name}: Skipped - {step_params['reason']}")
        
        steps['final'] = current_image
        
        processing_info = {
            'characteristics': characteristics,
            'strategy': strategy,
            'processing_log': processing_log,
            'steps': steps
        }
        
        if show_steps:
            self.visualize_adaptive_processing(steps, processing_info)
        
        print("  Adaptive preprocessing complete!")
        return current_image, processing_info
    
    def visualize_adaptive_processing(self, steps, processing_info):
        """Visualize the adaptive processing steps"""
        step_images = []
        step_titles = []
        
        for key, image in steps.items():
            if image is not None:
                step_images.append(image)
                if key == 'original':
                    step_titles.append('Original')
                elif key == 'final':
                    step_titles.append('Final Result')
                else:
                    step_titles.append(key.replace('after_', '').replace('_', ' ').title())
        
        n_steps = len(step_images)
        fig, axes = plt.subplots(2, max(3, (n_steps + 1) // 2), figsize=(16, 8))
        axes = axes.flatten()
        
        for i, (image, title) in enumerate(zip(step_images, step_titles)):
            if i < len(axes):
                axes[i].imshow(image, cmap='gray', vmin=0, vmax=255)
                axes[i].set_title(title, fontweight='bold')
                axes[i].axis('off')
                
                stats_text = f"Mean: {np.mean(image):.1f}\nStd: {np.std(image):.1f}"
                axes[i].text(0.02, 0.98, stats_text, transform=axes[i].transAxes,
                           verticalalignment='top', fontsize=8,
                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Add processing log
        if len(step_images) < len(axes):
            log_ax = axes[len(step_images)]
            log_ax.axis('off')
            
            log_text = "PROCESSING DECISIONS:\n\n"
            for entry in processing_info['processing_log']:
                log_text += f"• {entry}\n"
            
            log_ax.text(0.05, 0.95, log_text, transform=log_ax.transAxes,
                       verticalalignment='top', fontfamily='monospace', fontsize=9,
                       bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        # Hide unused subplots
        for i in range(len(step_images) + 1, len(axes)):
            axes[i].set_visible(False)
        
        plt.suptitle('Adaptive Preprocessing Pipeline', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()
    
    def process_dataset_adaptive(self, images_dir="Images"):
        """Apply adaptive preprocessing to entire dataset"""
        print("ADAPTIVE PREPROCESSING - DATASET PROCESSING")
        print("="*60)
        
        processed_data = self.dicom_handler.process_images_directory(images_dir)
        
        if not processed_data:
            print("No DICOM files could be processed.")
            return None
        
        results = []
        
        for i, (filename, original_image, metadata) in enumerate(processed_data):
            print(f"\nProcessing {i+1}/{len(processed_data)}: {filename}")
            
            processed_image, processing_info = self.process_image(original_image, show_steps=False)
            
            original_metrics = self.quality_analyzer.compute_comprehensive_metrics(original_image)
            processed_metrics = self.quality_analyzer.compute_comprehensive_metrics(processed_image)
            
            result = {
                'filename': filename,
                'original_image': original_image,
                'processed_image': processed_image,
                'original_metrics': original_metrics,
                'processed_metrics': processed_metrics,
                'processing_info': processing_info
            }
            
            results.append(result)
            
            # Print brief summary
            orig_quality = self._calculate_quality_score(original_metrics)
            proc_quality = self._calculate_quality_score(processed_metrics)
            improvement = proc_quality - orig_quality
            
            print(f"  Quality score: {orig_quality:.1f} → {proc_quality:.1f} ({improvement:+.1f})")
        
        self.analyze_adaptive_performance(results)
        return results
    
    def _calculate_quality_score(self, metrics):
        """Calculate overall quality score from metrics"""
        brightness = metrics['brightness']['mean_intensity']
        contrast = metrics['contrast']['rms_contrast']
        sharpness = metrics['sharpness']['laplacian_variance']
        snr = metrics['noise']['estimated_snr_db']
        
        # Normalize to 0-100 scale
        brightness_score = 100 - abs(brightness - 130) / 130 * 100
        contrast_score = min(100, contrast / 50 * 100)
        sharpness_score = min(100, sharpness / 500 * 100)
        snr_score = min(100, snr / 30 * 100)
        
        overall = (brightness_score + contrast_score + sharpness_score + snr_score) / 4
        return max(0, overall)
    
    def analyze_adaptive_performance(self, results):
        """Analyze adaptive preprocessing performance"""
        print("\n" + "="*60)
        print("ADAPTIVE PREPROCESSING PERFORMANCE ANALYSIS")
        print("="*60)
        
        improvements = 0
        total_improvement = 0
        
        print("\nIndividual Results:")
        print("-" * 40)
        
        for result in results:
            filename = result['filename']
            orig_quality = self._calculate_quality_score(result['original_metrics'])
            proc_quality = self._calculate_quality_score(result['processed_metrics'])
            improvement = proc_quality - orig_quality
            
            total_improvement += improvement
            if improvement > 0:
                improvements += 1
            
            status = "✓" if improvement > 0 else "✗" if improvement < -2 else "~"
            print(f"{filename:20} {orig_quality:5.1f} → {proc_quality:5.1f} ({improvement:+5.1f}) {status}")
        
        print("\nSummary:")
        print("-" * 20)
        print(f"Images improved: {improvements}/{len(results)} ({improvements/len(results)*100:.1f}%)")
        print(f"Average improvement: {total_improvement/len(results):+.1f} points")
        
        self.analyze_strategy_usage(results)
    
    def analyze_strategy_usage(self, results):
        """Analyze which processing strategies were used most often"""
        print(f"\nProcessing Strategy Usage:")
        print("-" * 30)
        
        strategy_counts = {process_type: {} for process_type in ['brightness', 'contrast', 'noise', 'sharpening']}
        
        for result in results:
            strategy = result['processing_info']['strategy']
            
            for process_type in strategy_counts.keys():
                method = strategy[process_type]['method']
                strategy_counts[process_type][method] = strategy_counts[process_type].get(method, 0) + 1
        
        for process_type, counts in strategy_counts.items():
            print(f"\n{process_type.title()} Methods:")
            for method, count in sorted(counts.items(), key=lambda x: x[1], reverse=True):
                print(f"  {method:20} {count:3d} times ({count/len(results)*100:4.1f}%)")