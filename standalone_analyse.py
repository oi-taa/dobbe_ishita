"""
Standalone Image Quality Analysis System
"""

from handle import DICOMHandler  
from analyze import ImageQualityAnalyzer
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

class StandaloneAnalysis:
    """
    Standalone analysis system
    """
    
    def __init__(self):
        self.dicom_handler = DICOMHandler()
        self.quality_analyzer = ImageQualityAnalyzer()
        self.analysis_results = {}
    
    def analyze_single_dicom(self, file_path, show_visualization=True):
        """
        Analyze a single DICOM file
        """
        print(f"Analyzing single DICOM: {file_path}")
        
        pixel_array, metadata = self.dicom_handler.read_dicom_file(file_path)
        
        if pixel_array is None:
            print("Failed to read DICOM file")
            return None
        
        # Perform quality analysis
        quality_metrics = self.quality_analyzer.compute_comprehensive_metrics(pixel_array)
        
        # Show visualization if requested
        if show_visualization:
            title = f"Quality Analysis: {file_path.split('/')[-1]}"
            self.quality_analyzer.visualize_quality_analysis(pixel_array, quality_metrics, title)
        
        # DICOM metadata + quality metrics
        combined_results = {
            'filename': file_path.split('/')[-1],
            'dicom_metadata': metadata,
            'quality_metrics': quality_metrics,
            'pixel_array': pixel_array
        }
        
        return combined_results
    
    def analyze_dataset_with_handler(self, images_dir="Images"):
        """
        Analyze entire dataset
        """
        print("DATASET ANALYSIS")
        print("="*60)
        
        processed_data = self.dicom_handler.process_images_directory(images_dir)
        
        if not processed_data:
            print("No images were processed successfully")
            return None
        
        print(f"\nAnalyzing quality metrics for {len(processed_data)} images...")
  
        all_results = []
        
        for i, (filename, pixel_array, metadata) in enumerate(processed_data):
            print(f"Quality analysis {i+1}/{len(processed_data)}: {filename}")
         
            quality_metrics = self.quality_analyzer.compute_comprehensive_metrics(pixel_array)
            
            result = {
                'filename': filename,
                'dicom_metadata': metadata,
                'quality_metrics': quality_metrics,
                'pixel_array': pixel_array
            }
            all_results.append(result)
        
        df = self._create_analysis_dataframe(all_results)
        
        self.analysis_results = {
            'dataframe': df,
            'detailed_results': all_results,
            'processed_data': processed_data
        }
    
        self._display_analysis_summary(df)
        
        self._plot_quality_distributions(df)
        
        return self.analysis_results
    
    def _create_analysis_dataframe(self, all_results):
        """
        Create a pandas DataFrame from analysis results
        """
        rows = []
        
        for result in all_results:
            row = {'filename': result['filename']}
  
            for key, value in result['dicom_metadata'].items():
                clean_key = key.replace(' ', '_').replace('(', '').replace(')', '').replace('/', '_')
                row[f"dicom_{clean_key}"] = value
            
            # Add quality metrics (flatten nested structure)
            for category, metrics in result['quality_metrics'].items():
                if isinstance(metrics, dict):
                    for metric_name, value in metrics.items():
                        row[f"{category}_{metric_name}"] = value
                else:
                    row[category] = metrics
            
            rows.append(row)
        
        return pd.DataFrame(rows)
    
    def _display_analysis_summary(self, df):
        """
        Display comprehensive analysis summary
        """
        print("\n" + "="*60)
        print("COMPREHENSIVE DATASET ANALYSIS SUMMARY")
        print("="*60)
        
        print(f"Total Images: {len(df)}")

        print(f"\nDICOM METADATA OVERVIEW:")
        if 'dicom_Equipment_Manufacturer' in df.columns:
            manufacturers = df['dicom_Equipment_Manufacturer'].value_counts()
            print(f"Equipment Manufacturers: {dict(manufacturers)}")
        
        if 'dicom_Image_Height' in df.columns and 'dicom_Image_Width' in df.columns:
            print(f"Image Dimensions:")
            print(f"  Height range: {df['dicom_Image_Height'].min()}-{df['dicom_Image_Height'].max()}")
            print(f"  Width range: {df['dicom_Image_Width'].min()}-{df['dicom_Image_Width'].max()}")
     
        print(f"\nQUALITY METRICS ANALYSIS:")
        
        key_metrics = [
            ('brightness_mean_intensity', 'Brightness', 80, 180),
            ('contrast_rms_contrast', 'Contrast', 30, 60),
            ('sharpness_laplacian_variance', 'Sharpness', 100, 500),
            ('noise_estimated_snr_db', 'Noise (SNR)', 20, 30)
        ]
        
        for metric, label, low_thresh, high_thresh in key_metrics:
            if metric in df.columns:
                values = df[metric]
                print(f"\n{label}:")
                print(f"  Range: {values.min():.2f} - {values.max():.2f}")
                print(f"  Mean ± Std: {values.mean():.2f} ± {values.std():.2f}")
                
                if metric == 'noise_estimated_snr_db':
                    poor = (values < low_thresh).sum()
                    good = (values > high_thresh).sum()
                else:
                    poor = ((values < low_thresh) | (values > high_thresh*2)).sum()
                    good = ((values >= low_thresh) & (values <= high_thresh*2)).sum()
                
                print(f"  Quality distribution: {good} good, {poor} poor, {len(df)-good-poor} moderate")
        
        print(f"\nIMAGES REQUIRING ATTENTION:")
        problematic = self._identify_problematic_images(df)
        
        if problematic:
            print(f"Found {len(problematic)} images with quality issues:")
            for filename, issues in problematic[:10]:  # Show first 10
                print(f"  {filename}: {', '.join(issues)}")
            if len(problematic) > 10:
                print(f"  ... and {len(problematic)-10} more")
        else:
            print("All images have acceptable quality metrics")
    
    def _identify_problematic_images(self, df):
        """
        Identify images with quality issues
        """
        problematic = []
        
        for _, row in df.iterrows():
            issues = []
         
            if 'brightness_mean_intensity' in df.columns:
                brightness = row['brightness_mean_intensity']
                if brightness < 80:
                    issues.append('too_dark')
                elif brightness > 180:
                    issues.append('too_bright')
            
            if 'contrast_rms_contrast' in df.columns:
                if row['contrast_rms_contrast'] < 30:
                    issues.append('low_contrast')
    
            if 'sharpness_laplacian_variance' in df.columns:
                if row['sharpness_laplacian_variance'] < 100:
                    issues.append('blurry')
         
            if 'noise_estimated_snr_db' in df.columns:
                if row['noise_estimated_snr_db'] < 20:
                    issues.append('noisy')
            
            if issues:
                problematic.append((row['filename'], issues))
        
        return problematic
    
    def _plot_quality_distributions(self, df):
        """
        Plot quality metric distributions
        """
        print("\nGenerating quality distribution plots...")
   
        metrics_to_plot = [
            ('brightness_mean_intensity', 'Mean Brightness'),
            ('contrast_rms_contrast', 'RMS Contrast'),
            ('sharpness_laplacian_variance', 'Laplacian Variance (Sharpness)'),
            ('noise_estimated_snr_db', 'Signal-to-Noise Ratio (dB)')
        ]
 
        available_metrics = [(col, title) for col, title in metrics_to_plot if col in df.columns]
        
        if not available_metrics:
            print("No quality metrics available for plotting")
            return

        n_metrics = len(available_metrics)
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()
        
        for i, (metric, title) in enumerate(available_metrics[:4]):
            ax = axes[i]
            values = df[metric].dropna()
            
            # Plot histogram
            ax.hist(values, bins=20, alpha=0.7, edgecolor='black', color='skyblue')
            ax.set_title(title)
            ax.set_xlabel('Value')
            ax.set_ylabel('Frequency')
            
            # Add statistics
            mean_val = values.mean()
            std_val = values.std()
            ax.axvline(mean_val, color='red', linestyle='--', linewidth=2,
                      label=f'Mean: {mean_val:.1f}')
            ax.axvline(mean_val + std_val, color='orange', linestyle=':', alpha=0.7)
            ax.axvline(mean_val - std_val, color='orange', linestyle=':', alpha=0.7)
            ax.legend()
            ax.grid(True, alpha=0.3)
       
        for i in range(len(available_metrics), 4):
            axes[i].set_visible(False)
        
        plt.suptitle('Dataset Quality Metrics Distribution', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()
    
    def generate_preprocessing_recommendations(self):
        """
        Generate preprocessing recommendations based on analysis
        """
        if not self.analysis_results:
            print("Please run analyze_dataset_with_handler() first")
            return
        
        df = self.analysis_results['dataframe']
        
        print("\n" + "="*60)
        print("PREPROCESSING RECOMMENDATIONS")
        print("="*60)
        
        recommendations = {}
        
        for _, row in df.iterrows():
            filename = row['filename']
            recs = []
   
            if 'brightness_mean_intensity' in df.columns:
                brightness = row['brightness_mean_intensity']
                if brightness < 80:
                    recs.append("Apply gamma correction (γ > 1) to brighten")
                elif brightness > 180:
                    recs.append("Apply gamma correction (γ < 1) to darken")
            
            if 'contrast_rms_contrast' in df.columns:
                if row['contrast_rms_contrast'] < 30:
                    recs.append("Apply CLAHE or histogram equalization")
            
            if 'sharpness_laplacian_variance' in df.columns:
                if row['sharpness_laplacian_variance'] < 100:
                    recs.append("Apply unsharp masking or Laplacian sharpening")
 
            if 'noise_estimated_snr_db' in df.columns:
                if row['noise_estimated_snr_db'] < 20:
                    recs.append("Apply Gaussian blur or median filtering")
            
            recommendations[filename] = recs
 
        needs_preprocessing = {k: v for k, v in recommendations.items() if v}
        
        print(f"Images needing preprocessing: {len(needs_preprocessing)}/{len(df)}")
        print(f"Images with good quality: {len(df) - len(needs_preprocessing)}/{len(df)}")
        
        if needs_preprocessing:
            print("\nPreprocessing recommendations:")
            for filename, recs in needs_preprocessing.items():
                print(f"\n{filename}:")
                for rec in recs:
                    print(f"  • {rec}")
        
        return recommendations
    
    def save_analysis_results(self, output_prefix="dataset_analysis"):
        """
        Save analysis results to CSV files
        """
        if not self.analysis_results:
            print("No analysis results to save")
            return
        
        df = self.analysis_results['dataframe']
        
        filename = f"{output_prefix}_detailed.csv"
        df.to_csv(filename, index=False)
        print(f"✓ Detailed analysis saved to: {filename}")
        
        summary_data = []
        
        quality_columns = [col for col in df.columns if any(
            col.startswith(prefix) for prefix in ['brightness_', 'contrast_', 'sharpness_', 'noise_']
        )]
        
        for col in quality_columns:
            values = df[col].dropna()
            if len(values) > 0:
                summary_data.append({
                    'metric': col,
                    'count': len(values),
                    'mean': values.mean(),
                    'std': values.std(),
                    'min': values.min(),
                    'max': values.max(),
                    'q25': values.quantile(0.25),
                    'q50': values.quantile(0.50),
                    'q75': values.quantile(0.75)
                })
        
        summary_df = pd.DataFrame(summary_data)
        summary_filename = f"{output_prefix}_summary.csv"
        summary_df.to_csv(summary_filename, index=False)
        print(f"✓ Summary statistics saved to: {summary_filename}")
    
    def compare_images_by_filename(self, filenames, show_detailed=True):
        """
        Compare specific images by filename
        """
        if not self.analysis_results:
            print("Please run dataset analysis first")
            return
        
        df = self.analysis_results['dataframe']
        detailed_results = self.analysis_results['detailed_results']
        
        print(f"\nCOMPARING {len(filenames)} IMAGES:")
        print("="*50)

        comparison_data = []
        for filename in filenames:
            row = df[df['filename'] == filename]
            if row.empty:
                print(f"Warning: {filename} not found in analysis results")
                continue
            
            detailed = next((r for r in detailed_results if r['filename'] == filename), None)
            if detailed:
                comparison_data.append((filename, row.iloc[0], detailed))
        
        if not comparison_data:
            print("No valid images found for comparison")
            return
        
        metrics = ['brightness_mean_intensity', 'contrast_rms_contrast', 
                  'sharpness_laplacian_variance', 'noise_estimated_snr_db']
        
        print(f"{'Metric':<25}", end='')
        for filename, _, _ in comparison_data:
            print(f"{filename[:15]:<18}", end='')
        print()
        print("-" * (25 + 18 * len(comparison_data)))
        
        for metric in metrics:
            if metric in df.columns:
                print(f"{metric.replace('_', ' '):<25}", end='')
                for _, row, _ in comparison_data:
                    value = row[metric]
                    print(f"{value:<18.2f}", end='')
                print()
        
        if show_detailed:
            for filename, row, detailed in comparison_data:
                print(f"\nDetailed analysis for: {filename}")
                self.quality_analyzer.visualize_quality_analysis(
                    detailed['pixel_array'], 
                    detailed['quality_metrics'], 
                    f"Analysis: {filename}"
                )

