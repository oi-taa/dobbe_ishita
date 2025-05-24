"""
Static Preprocessing Pipeline- baseline
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
from handle import DICOMHandler
from analyze import ImageQualityAnalyzer

class StaticPreprocessor:
    """
    Static preprocessing pipeline with fixed parameters for all images
    """
    
    def __init__(self):
        self.dicom_handler = DICOMHandler()
        self.quality_analyzer = ImageQualityAnalyzer()
        
        self.brightness_gamma = 1.2  # Fixed gamma correction
        self.contrast_clip_limit = 1.2  # Fixed CLAHE parameters
        self.contrast_grid_size = (8, 8)
        self.sharpen_strength = 1.5  # Fixed sharpening
        self.denoise_kernel_size = 3  # Fixed denoising
        self.denoise_sigma = 1.0
    
    def apply_brightness_correction(self, image):
        """
        Apply fixed gamma correction for brightness
        """
        normalized = image.astype(np.float32) / 255.0
        
        # Apply fixed gamma correction
        corrected = np.power(normalized, 1.0 / self.brightness_gamma)
        
        # Convert back to 0-255 range
        result = (corrected * 255).astype(np.uint8)
        
        return result
    
    def apply_contrast_enhancement(self, image):
        """
        Apply fixed CLAHE (Contrast Limited Adaptive Histogram Equalization)
        """
        # Create CLAHE object with fixed parameters
        clahe = cv2.createCLAHE(
            clipLimit=self.contrast_clip_limit,
            tileGridSize=self.contrast_grid_size
        )

        enhanced = clahe.apply(image)
        
        return enhanced
    
    def apply_sharpening(self, image):
        """
        Apply fixed unsharp masking for sharpening
        """
        blurred = cv2.GaussianBlur(image, (5, 5), 1.0)
        
        # Create unsharp mask
        unsharp_mask = cv2.addWeighted(
            image, 1.0 + self.sharpen_strength,
            blurred, -self.sharpen_strength,
            0
        )
        
        # Clip values to valid range
        sharpened = np.clip(unsharp_mask, 0, 255).astype(np.uint8)
        
        return sharpened
    
    def apply_denoising(self, image):
        """
        Apply fixed Gaussian denoising
        """
        denoised = cv2.GaussianBlur(
            image, 
            (self.denoise_kernel_size, self.denoise_kernel_size), 
            self.denoise_sigma
        )
        
        return denoised
    
    def static_preprocessing_pipeline(self, image, show_steps=False):
        """
        Apply the complete static preprocessing pipeline
        """
        steps = {'original': image.copy()}
        
        # Step 1: Brightness correction (fixed gamma)
        print("  Step 1: Applying fixed brightness correction (γ=1.2)...")
        brightness_corrected = self.apply_brightness_correction(image)
        steps['brightness_corrected'] = brightness_corrected
        
        # Step 2: Contrast enhancement (fixed CLAHE)
        print("  Step 2: Applying fixed contrast enhancement (CLAHE)...")
        contrast_enhanced = self.apply_contrast_enhancement(brightness_corrected)
        steps['contrast_enhanced'] = contrast_enhanced
        
        # Step 3: Sharpening (fixed unsharp masking)
        print("  Step 3: Applying fixed sharpening (strength=1.5)...")
        sharpened = self.apply_sharpening(contrast_enhanced)
        steps['sharpened'] = sharpened
        
        # Step 4: Denoising (fixed Gaussian blur)
        print("  Step 4: Applying fixed denoising (σ=1.0)...")
        denoised = self.apply_denoising(sharpened)
        steps['final'] = denoised
        
        if show_steps:
            self.visualize_preprocessing_steps(steps)
        
        return denoised, steps
    
    def visualize_preprocessing_steps(self, steps):
        """
        Visualize each step of the preprocessing pipeline
        """
        step_names = ['original', 'brightness_corrected', 'contrast_enhanced', 'sharpened', 'final']
        step_titles = ['Original', 'Brightness\nCorrected', 'Contrast\nEnhanced', 'Sharpened', 'Final\nResult']
        
        fig, axes = plt.subplots(1, 5, figsize=(20, 4))
        
        for i, (step_name, title) in enumerate(zip(step_names, step_titles)):
            if step_name in steps:
                axes[i].imshow(steps[step_name], cmap='gray', vmin=0, vmax=255)
                axes[i].set_title(title, fontsize=12, fontweight='bold')
                axes[i].axis('off')
                
                # Add basic statistics
                img = steps[step_name]
                stats_text = f"Mean: {np.mean(img):.1f}\nStd: {np.std(img):.1f}"
                axes[i].text(0.02, 0.98, stats_text, transform=axes[i].transAxes,
                           verticalalignment='top', fontsize=8,
                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.suptitle('Static Preprocessing Pipeline Steps', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()
    
    def process_dataset_static(self, images_dir="Images"):
        """
        Apply static preprocessing to entire dataset and analyze results
        """
        print("STATIC PREPROCESSING ANALYSIS")
        print("="*50)
        print("Applying fixed preprocessing pipeline to all images...")
        print(f"Pipeline: γ={self.brightness_gamma} → CLAHE({self.contrast_clip_limit}) → ")
        print(f"         Sharpen({self.sharpen_strength}) → Denoise(σ={self.denoise_sigma})")
        print()
        
        processed_data = self.dicom_handler.process_images_directory(images_dir)
        
        if not processed_data:
            print("No DICOM files could be processed.")
            return None
        
        results = []
        
        for i, (filename, original_image, metadata) in enumerate(processed_data):
            print(f"\nProcessing {i+1}/{len(processed_data)}: {filename}")
            
            original_metrics = self.quality_analyzer.compute_comprehensive_metrics(original_image)
            
            processed_image, steps = self.static_preprocessing_pipeline(original_image, show_steps=False)
            
            processed_metrics = self.quality_analyzer.compute_comprehensive_metrics(processed_image)
            
            result = {
                'filename': filename,
                'original_image': original_image,
                'processed_image': processed_image,
                'original_metrics': original_metrics,
                'processed_metrics': processed_metrics,
                'steps': steps
            }
            results.append(result)
        
        self.analyze_static_performance(results)
        
        return results
    
    def analyze_static_performance(self, results):
        """
        Analyze how well static preprocessing performed across the dataset
        """
        print("\n" + "="*60)
        print("STATIC PREPROCESSING PERFORMANCE ANALYSIS")
        print("="*60)

        original_data = []
        processed_data = []
        
        for result in results:
            orig_row = {'filename': result['filename']}
            for category, metrics in result['original_metrics'].items():
                if isinstance(metrics, dict):
                    for metric_name, value in metrics.items():
                        orig_row[f"{category}_{metric_name}"] = value
            original_data.append(orig_row)
            
            # Processed metrics
            proc_row = {'filename': result['filename']}
            for category, metrics in result['processed_metrics'].items():
                if isinstance(metrics, dict):
                    for metric_name, value in metrics.items():
                        proc_row[f"{category}_{metric_name}"] = value
            processed_data.append(proc_row)
        
        original_df = pd.DataFrame(original_data)
        processed_df = pd.DataFrame(processed_data)
        
        key_metrics = [
            'brightness_mean_intensity',
            'contrast_rms_contrast',
            'sharpness_laplacian_variance',
            'noise_estimated_snr_db'
        ]
        
        print("\nOVERALL PERFORMANCE SUMMARY:")
        print("-" * 40)
        
        improvements = 0
        deteriorations = 0
        
        for metric in key_metrics:
            if metric in original_df.columns and metric in processed_df.columns:
                orig_mean = original_df[metric].mean()
                proc_mean = processed_df[metric].mean()
                
                if metric == 'noise_estimated_snr_db':
                    # Higher SNR is better
                    improved = proc_mean > orig_mean
                    change = proc_mean - orig_mean
                    change_pct = (change / orig_mean) * 100
                elif metric == 'brightness_mean_intensity':
                    # Target range 80-180, check if we moved closer
                    orig_dist = min(abs(orig_mean - 80), abs(orig_mean - 180))
                    proc_dist = min(abs(proc_mean - 80), abs(proc_mean - 180))
                    improved = proc_dist < orig_dist
                    change = proc_mean - orig_mean
                    change_pct = (change / orig_mean) * 100
                else:
                    # Higher contrast and sharpness generally better
                    improved = proc_mean > orig_mean
                    change = proc_mean - orig_mean
                    change_pct = (change / orig_mean) * 100
                
                if improved:
                    improvements += 1
                    status = "✓ IMPROVED"
                else:
                    deteriorations += 1
                    status = "✗ DETERIORATED"
                
                print(f"{metric.replace('_', ' ').title()}:")
                print(f"  Original: {orig_mean:.2f}")
                print(f"  Processed: {proc_mean:.2f}")
                print(f"  Change: {change:+.2f} ({change_pct:+.1f}%) {status}")
                print()
        
        print(f"Summary: {improvements} metrics improved, {deteriorations} metrics deteriorated")
        
        self.analyze_individual_performance(results, original_df, processed_df)
        
        self.identify_failure_cases(results, original_df, processed_df)
    
    def analyze_individual_performance(self, results, original_df, processed_df):
        """
        Analyze performance on individual images to show variability
        """
        print("\nINDIVIDUAL IMAGE PERFORMANCE:")
        print("-" * 40)
        
        # Calculate improvement scores for each image
        for i, result in enumerate(results):
            filename = result['filename']
            
            orig_metrics = original_df.iloc[i]
            proc_metrics = processed_df.iloc[i]
            
            improvements = 0
            total_metrics = 0
            
            key_metrics = ['brightness_mean_intensity', 'contrast_rms_contrast', 
                          'sharpness_laplacian_variance', 'noise_estimated_snr_db']
            
            for metric in key_metrics:
                if metric in orig_metrics and metric in proc_metrics:
                    total_metrics += 1
                    orig_val = orig_metrics[metric]
                    proc_val = proc_metrics[metric]
                    
                    if metric == 'noise_estimated_snr_db':
                        if proc_val > orig_val:
                            improvements += 1
                    elif metric == 'brightness_mean_intensity':
                        # Check if moved closer to optimal range (80-180)
                        orig_dist = min(abs(orig_val - 80), abs(orig_val - 180), abs(orig_val - 130))
                        proc_dist = min(abs(proc_val - 80), abs(proc_val - 180), abs(proc_val - 130))
                        if proc_dist < orig_dist:
                            improvements += 1
                    else:
                        if proc_val > orig_val:
                            improvements += 1
            
            success_rate = (improvements / total_metrics) * 100 if total_metrics > 0 else 0
            
            if success_rate >= 75:
                status = "✓ GOOD"
            elif success_rate >= 50:
                status = "~ MIXED"
            else:
                status = "✗ POOR"
            
            print(f"{filename}: {improvements}/{total_metrics} metrics improved ({success_rate:.0f}%) {status}")
    
    def identify_failure_cases(self, results, original_df, processed_df):
        """
        Identify and analyze specific failure cases
        """
        print("\n" + "="*60)
        print("FAILURE CASE ANALYSIS")
        print("="*60)
        
        failure_cases = []
        
        for i, result in enumerate(results):
            filename = result['filename']
            orig_metrics = original_df.iloc[i]
            proc_metrics = processed_df.iloc[i]
            
            failures = []
            
            # Brightness failures
            orig_brightness = orig_metrics['brightness_mean_intensity']
            proc_brightness = proc_metrics['brightness_mean_intensity']
            
            if orig_brightness < 80 and proc_brightness < 80:
                failures.append("Failed to brighten dark image")
            elif orig_brightness > 180 and proc_brightness > 180:
                failures.append("Failed to darken bright image")
            elif 80 <= orig_brightness <= 180 and (proc_brightness < 70 or proc_brightness > 190):
                failures.append("Ruined good brightness")
            
            # Contrast failures
            orig_contrast = orig_metrics['contrast_rms_contrast']
            proc_contrast = proc_metrics['contrast_rms_contrast']
            
            if orig_contrast < 30 and proc_contrast < 35:
                failures.append("Failed to improve low contrast")
            elif orig_contrast > 50 and proc_contrast < orig_contrast * 0.8:
                failures.append("Reduced good contrast")
            
            # Sharpness failures
            orig_sharpness = orig_metrics['sharpness_laplacian_variance']
            proc_sharpness = proc_metrics['sharpness_laplacian_variance']
            
            if proc_sharpness < orig_sharpness * 0.7:
                failures.append("Over-smoothed (lost sharpness)")
            
            # Noise failures
            orig_snr = orig_metrics['noise_estimated_snr_db']
            proc_snr = proc_metrics['noise_estimated_snr_db']
            
            if orig_snr > 25 and proc_snr < orig_snr - 3:
                failures.append("Introduced noise to clean image")
            
            if failures:
                failure_cases.append((filename, failures, orig_metrics, proc_metrics))
        
        if failure_cases:
            print(f"Found {len(failure_cases)} images with significant failures:")
            print()
            
            for filename, failures, orig, proc in failure_cases[:5]:  # Show first 5
                print(f"📁 {filename}:")
                for failure in failures:
                    print(f"   • {failure}")
                
                print(f"   Metrics comparison:")
                print(f"     Brightness: {orig['brightness_mean_intensity']:.1f} → {proc['brightness_mean_intensity']:.1f}")
                print(f"     Contrast: {orig['contrast_rms_contrast']:.1f} → {proc['contrast_rms_contrast']:.1f}")
                print(f"     Sharpness: {orig['sharpness_laplacian_variance']:.1f} → {proc['sharpness_laplacian_variance']:.1f}")
                print(f"     SNR: {orig['noise_estimated_snr_db']:.1f} → {proc['noise_estimated_snr_db']:.1f}")
                print()
            
            if len(failure_cases) > 5:
                print(f"   ... and {len(failure_cases) - 5} more failure cases")
        else:
            print("No major failure cases detected.")
        
        # To show why static preprocessing fails
        self.explain_static_failures(failure_cases)
    
    def explain_static_failures(self, failure_cases):
        """
        Explain why the static approach fails
        """
        print("\n" + "="*60)
        print("WHY STATIC PREPROCESSING FAILS")
        print("="*60)
        
        explanations = [
            "🔧 FIXED PARAMETERS PROBLEM:",
            "   • One gamma value (1.2) can't handle both dark AND bright images",
            "   • Fixed CLAHE parameters over-enhance good images, under-enhance poor ones",
            "   • Fixed sharpening strength blurs noisy images, under-sharpens blurry ones",
            "   • Fixed denoising removes detail from sharp images, inadequate for noisy ones",
            "",
            "📊 DATASET DIVERSITY CHALLENGE:",
            "   • Images have different acquisition conditions",
            "   • Different exposure levels require different brightness corrections",
            "   • Different noise levels need adaptive denoising strategies",
            "   • Different blur levels need varying sharpening approaches",
            "",
            "⚖️ ONE-SIZE-FITS-ALL LIMITATIONS:",
            "   • Optimal γ for dark image = 0.7, for bright image = 1.5",
            "   • Low contrast images need strong CLAHE, high contrast images need mild",
            "   • Noisy images need denoising first, clean images need sharpening",
            "   • Can't optimize for contradictory requirements simultaneously",
            "",
            "🎯 SOLUTION NEEDED:",
            "   • ADAPTIVE preprocessing that analyzes each image first",
            "   • Different parameters based on measured image characteristics",
            "   • Conditional processing pipelines",
            "   • Quality-driven parameter selection"
        ]
        
        for explanation in explanations:
            print(explanation)
        
        print("\n" + "="*60)
        print("CONCLUSION: Static preprocessing fails because dental X-rays")
        print("have too much variability for fixed parameters to work well.")
        print("Next step: Implement ADAPTIVE preprocessing pipeline!")
        print("="*60)
    
    def visualize_failure_examples(self, results, num_examples=3):
        """
        Visualize specific examples where static preprocessing failed
        """
        print(f"\nVisualizing {num_examples} failure examples...")
        
        # Find worst performing images
        worst_images = []
        
        for result in results:
            filename = result['filename']
            orig = result['original_metrics']
            proc = result['processed_metrics']
            
            # Calculate degradation score
            degradation = 0
            
            # Brightness degradation
            orig_brightness = orig['brightness']['mean_intensity']
            proc_brightness = proc['brightness']['mean_intensity']
            if 80 <= orig_brightness <= 180:  # Was good
                if proc_brightness < 70 or proc_brightness > 190:  # Now bad
                    degradation += 2
            
            # Sharpness loss
            orig_sharpness = orig['sharpness']['laplacian_variance']
            proc_sharpness = proc['sharpness']['laplacian_variance']
            if proc_sharpness < orig_sharpness * 0.7:
                degradation += 1
            
            # SNR loss
            orig_snr = orig['noise']['estimated_snr_db']
            proc_snr = proc['noise']['estimated_snr_db']
            if proc_snr < orig_snr - 2:
                degradation += 1
            
            worst_images.append((degradation, result))
        
        # Sort by degradation and take worst examples
        worst_images.sort(key=lambda x: x[0], reverse=True)
        
        for i, (degradation_score, result) in enumerate(worst_images[:num_examples]):
            print(f"\nFailure Example {i+1}: {result['filename']} (degradation score: {degradation_score})")
            
            # Show before/after comparison
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            
            # Original
            axes[0].imshow(result['original_image'], cmap='gray', vmin=0, vmax=255)
            axes[0].set_title('Original Image')
            axes[0].axis('off')
            
            # Processed
            axes[1].imshow(result['processed_image'], cmap='gray', vmin=0, vmax=255)
            axes[1].set_title('After Static Preprocessing')
            axes[1].axis('off')
            
            plt.suptitle(f'Static Preprocessing Failure: {result["filename"]}', 
                        fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.show()
            
            # Print metric comparison
            orig = result['original_metrics']
            proc = result['processed_metrics']
            
            print("Metric Comparison:")
            print(f"  Brightness: {orig['brightness']['mean_intensity']:.1f} → {proc['brightness']['mean_intensity']:.1f}")
            print(f"  Contrast: {orig['contrast']['rms_contrast']:.1f} → {proc['contrast']['rms_contrast']:.1f}")
            print(f"  Sharpness: {orig['sharpness']['laplacian_variance']:.1f} → {proc['sharpness']['laplacian_variance']:.1f}")
            print(f"  SNR: {orig['noise']['estimated_snr_db']:.1f} → {proc['noise']['estimated_snr_db']:.1f}")

def demo_static_preprocessing():
    """
    Demonstrate static preprocessing and its limitations
    """
    print("STATIC PREPROCESSING DEMONSTRATION")
    print("="*50)
    print("This demo shows why fixed preprocessing parameters fail")
    print("on diverse dental X-ray datasets.")
    print()

    preprocessor = StaticPreprocessor()
    
    results = preprocessor.process_dataset_static("Images")
    
    if results:
        preprocessor.visualize_failure_examples(results, num_examples=2)
        
        print("\n" + "="*60)
        print("DEMONSTRATION COMPLETE")
        print("="*60)
        print("Key takeaways:")
        print("1. Fixed parameters cannot handle image diversity")
        print("2. Static preprocessing often makes images worse")
        print("3. Different images need different processing strategies")
        print("4. Adaptive preprocessing is essential for optimal results")
        print()

if __name__ == "__main__":
    demo_static_preprocessing()