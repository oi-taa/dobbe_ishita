import pydicom
import numpy as np
import matplotlib.pyplot as plt
import cv2
from pathlib import Path

class DICOMHandler:
    """DICOM file handler for dental X-rays"""
    
    def __init__(self):
        self.current_image = None
        self.current_metadata = {}
    
    def read_dicom_file(self, file_path):
        """
        extract pixel data + metadata
        """
        try:
            print(f"Reading DICOM file: {file_path}")
            
            dicom_data = pydicom.dcmread(file_path, force=True)
            
            # Extract pixel array -0 - black 255 - white
            if hasattr(dicom_data, 'pixel_array'):
                pixel_array = dicom_data.pixel_array
                
                # Handle different bit depths - normalizing to 8-bit range(255)
                if pixel_array.dtype != np.uint8:
                    pixel_min = np.min(pixel_array)
                    pixel_max = np.max(pixel_array)
                    if pixel_max > pixel_min:
                        pixel_array = ((pixel_array - pixel_min) / (pixel_max - pixel_min) * 255).astype(np.uint8)
                    else:
                        pixel_array = np.zeros_like(pixel_array, dtype=np.uint8)
                
                # If Monochrome 1 then need to reverse to behave like Monochrome 2
                if hasattr(dicom_data, 'PhotometricInterpretation'):
                    if dicom_data.PhotometricInterpretation == 'MONOCHROME1':
                        pixel_array = 255 - pixel_array
                
                print(f"✓ Successfully extracted pixel data: {pixel_array.shape}, dtype: {pixel_array.dtype}")
                
            else:
                raise ValueError("No pixel data found in DICOM file")
         
            metadata = self._extract_metadata(dicom_data)
            
            self.current_image = pixel_array
            self.current_metadata = metadata
            
            return pixel_array, metadata
            
        except Exception as e:
            print(f"✗ Error reading DICOM file: {str(e)}")
            return None, None
    
    def _extract_metadata(self, dicom_data):
        """
        Extract metadata from the DICOM object
        """
        metadata = {}

        # Essential DICOM tags for dental X-rays - considering how these tags affect processing of the DICOm img, how the Xray was taken, and optimal appearance of the img
        essential_tags = {
            # Core image properties (all present)
            'PatientID': 'Patient ID',
            'StudyDate': 'Study Date',
            'Modality': 'Imaging Modality',
            'PhotometricInterpretation': 'Photometric Interpretation',
            'Rows': 'Image Height',
            'Columns': 'Image Width',
            'BitsAllocated': 'Bits Allocated',
            'BitsStored': 'Bits Stored',
            'PixelSpacing': 'Pixel Spacing (mm)',
            
            # Equipment identification (for adaptive strategies)
            'Manufacturer': 'Equipment Manufacturer',
            'ManufacturerModelName': 'Equipment Model',
            'DeviceSerialNumber': 'Device Serial Number',
            'SoftwareVersions': 'Software Version',
            
            # Processing parameters  
            'SpatialResolution': 'Spatial Resolution',
            'DetectorType': 'Detector Type',
            'RescaleIntercept': 'Rescale Intercept',
            'RescaleSlope': 'Rescale Slope',
            
            # Orientation/positioning
            'FieldOfViewRotation': 'Field of View Rotation',
            'ImageLaterality': 'Image Side',
            
            # Image classification
            'ImageType': 'Image Type',
        }

        for tag, description in essential_tags.items():
            if hasattr(dicom_data, tag):
                value = getattr(dicom_data, tag)
                metadata[description] = value
        
        # Add computed metrics
        if self.current_image is not None:
            metadata['Computed Image Shape'] = self.current_image.shape
            metadata['Computed Data Type'] = str(self.current_image.dtype)
            metadata['Computed Min Value'] = int(np.min(self.current_image))
            metadata['Computed Max Value'] = int(np.max(self.current_image))
            metadata['Computed Mean Value'] = f"{np.mean(self.current_image):.2f}"
            metadata['Computed Std Value'] = f"{np.std(self.current_image):.2f}"
        
        return metadata
    
    def visualize_dicom_image(self, pixel_array=None, title="DICOM X-ray Image"):
        """
        Display DICOM image with histogram and basic analysis
        """
        if pixel_array is None:
            pixel_array = self.current_image
        
        if pixel_array is None:
            print("No image data to visualize")
            return
        
        # fig with img and histogram
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Xray
        ax1.imshow(pixel_array, cmap='gray', vmin=0, vmax=255)
        ax1.set_title(f'{title}\nSize: {pixel_array.shape}')
        ax1.axis('off')
        
        # Add image statistics as text overlay
        stats_text = f"Min: {np.min(pixel_array)}\nMax: {np.max(pixel_array)}\nMean: {np.mean(pixel_array):.1f}\nStd: {np.std(pixel_array):.1f}"
        ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                fontsize=10, fontfamily='monospace')
        
        ax2.hist(pixel_array.flatten(), bins=50, alpha=0.7, color='blue', edgecolor='black')
        ax2.set_title('Pixel Intensity Distribution')
        ax2.set_xlabel('Pixel Intensity (0-255)')
        ax2.set_ylabel('Frequency')
        ax2.grid(True, alpha=0.3)
        
        mean_val = np.mean(pixel_array)
        std_val = np.std(pixel_array)
        median_val = np.median(pixel_array)
        
        ax2.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.1f}')
        ax2.axvline(median_val, color='green', linestyle='--', linewidth=2, label=f'Median: {median_val:.1f}')
        ax2.axvline(mean_val + std_val, color='orange', linestyle=':', alpha=0.7, label=f'±1 STD')
        ax2.axvline(mean_val - std_val, color='orange', linestyle=':', alpha=0.7)
        ax2.legend()
        
        plt.tight_layout()
        plt.show()
    
    def print_metadata(self, metadata=None):
        """
        Print formatted metadata information
        """
        if metadata is None:
            metadata = self.current_metadata
        
        if not metadata:
            print("No metadata available")
            return
        
        print("\n" + "="*60)
        print("DICOM METADATA INFORMATION")
        print("="*60)
  
        dicom_info = {}     
        imaging_info = {}   # Keys containing specific technical terms:
        computed_info = {}  # Keys containing 'computed'
        
        for key, value in metadata.items():
            if key.startswith('Computed'):
                computed_info[key] = value
            elif 'Exposure' in key or 'Current' in key or 'Voltage' in key or 'Window' in key:
                imaging_info[key] = value
            else:
                dicom_info[key] = value
        
        # Print categorized information
        if dicom_info:
            print("DICOM FILE INFORMATION:")
            for key, value in dicom_info.items():
                print(f"  {key:<35}: {value}")
        
        if imaging_info:
            print(f"\nIMAGING PARAMETERS:")
            for key, value in imaging_info.items():
                print(f"  {key:<35}: {value}")
        
        if computed_info:
            print(f"\nCOMPUTED STATISTICS:")
            for key, value in computed_info.items():
                print(f"  {key:<35}: {value}")
        
        print("="*60)
    
    def process_images_directory(self, images_dir="Images"):
        """
        Process all DICOM files in the Images directory
        """
        images_path = Path(images_dir)
        
        if not images_path.exists():
            print(f"✗ Images directory '{images_dir}' not found!")
            print(f"   Current working directory: {Path.cwd()}")
            print(f"   Looking for: {images_path.absolute()}")
            return []
        
        dicom_extensions = ['*.dcm', '*.dicom', '*.DCM', '*.DICOM', '*.rvg', '*.RVG']
        dicom_files = set()  # to avoid duplicates
        
        for ext in dicom_extensions:
            dicom_files.update(images_path.glob(ext))
        
        for file_path in images_path.iterdir():
            if file_path.is_file() and not file_path.suffix:
                dicom_files.add(file_path)
        
        dicom_files = list(dicom_files) 
        
        print(f"Found {len(dicom_files)} potential DICOM files in '{images_dir}'")
        
        if len(dicom_files) == 0:
            print("No DICOM files found. Files in directory:")
            for file_path in images_path.iterdir():
                print(f"  - {file_path.name}")
            return []

        processed_data = []
        successful = 0
        
        for file_path in sorted(dicom_files): 
            print(f"\nProcessing: {file_path.name}")
            pixel_array, metadata = self.read_dicom_file(str(file_path))
            
            if pixel_array is not None:
                processed_data.append((file_path.name, pixel_array, metadata))
                successful += 1
                print(f"  ✓ Success - Shape: {pixel_array.shape}, Range: {pixel_array.min()}-{pixel_array.max()}")
            else:
                print(f"  ✗ Failed to process {file_path.name}")
        
        print(f"\n{'='*60}")
        print(f"PROCESSING SUMMARY: {successful}/{len(dicom_files)} files processed successfully")
        print(f"{'='*60}")
        
        return processed_data

