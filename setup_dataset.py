#!/usr/bin/env python3
"""
Deepfake Detection Dataset Setup Script
Combines dataset downloading and creation functionality
"""

import os
import shutil
from pathlib import Path
import requests
import zipfile
import tarfile
from typing import List, Optional
import random

try:
    import kagglehub
    KAGGLE_AVAILABLE = True
except ImportError:
    KAGGLE_AVAILABLE = False
    print("Warning: kagglehub not available. Install with: pip install kagglehub")

class DatasetSetup:
    """Handles downloading and organizing deepfake detection datasets"""
    
    def __init__(self, base_dir: str = "datasets"):
        self.base_dir = Path(base_dir)
        self.setup_directories()
    
    def setup_directories(self):
        """Create necessary directory structure"""
        dirs = [
            self.base_dir / "train" / "real",
            self.base_dir / "train" / "fake", 
            self.base_dir / "val" / "real",
            self.base_dir / "val" / "fake",
            "downloads"
        ]
        
        for dir_path in dirs:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
        
        print(f"Directory structure created in {self.base_dir}")
    
    def download_kaggle_datasets(self):
        """Download datasets from Kaggle using kagglehub"""
        if not KAGGLE_AVAILABLE:
            print("Error: kagglehub not available. Install with: pip install kagglehub")
            return False
        
        print("Downloading datasets from Kaggle...")
        
        try:
            # Download FaceForensics++ C23 dataset
            print("Downloading FaceForensics++ C23 dataset...")
            path_FFc23 = kagglehub.dataset_download(
                "gradientvoyager/faceforensics-c23-extracted-faces-100k"
            )
            print(f"FaceForensics++ C23 downloaded to: {path_FFc23}")
            
            # Download Tiny GenImage dataset
            print("Downloading Tiny GenImage dataset...")
            path_GenImage = kagglehub.dataset_download("cartografia/unbiased-tiny-genimage")
            print(f"Tiny GenImage downloaded to: {path_GenImage}")
            
            return {
                "faceforensics": path_FFc23,
                "genimage": path_GenImage
            }
            
        except Exception as e:
            print(f"Error downloading datasets: {e}")
            return False
    
    def download_file(self, url: str, destination: str, description: str = "file") -> bool:
        """Download file with progress indication"""
        try:
            print(f"Downloading {description}...")
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            downloaded = 0
            
            with open(destination, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total_size > 0:
                            progress = (downloaded / total_size) * 100
                            print(f"\rProgress: {progress:.1f}%", end='', flush=True)
            
            print(f"\n{description} downloaded successfully")
            return True
            
        except Exception as e:
            print(f"Failed to download {description}: {e}")
            return False
    
    def extract_archive(self, archive_path: str, extract_to: str) -> bool:
        """Extract zip or tar archives"""
        try:
            print(f"Extracting {archive_path}...")
            
            if archive_path.endswith('.zip'):
                with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                    zip_ref.extractall(extract_to)
            elif archive_path.endswith(('.tar.gz', '.tgz')):
                with tarfile.open(archive_path, 'r:gz') as tar_ref:
                    tar_ref.extractall(extract_to)
            elif archive_path.endswith('.tar'):
                with tarfile.open(archive_path, 'r') as tar_ref:
                    tar_ref.extractall(extract_to)
            else:
                print(f"Unsupported archive format: {archive_path}")
                return False
            
            print(f"Extracted to {extract_to}")
            return True
            
        except Exception as e:
            print(f"Failed to extract {archive_path}: {e}")
            return False
    
    def setup_faceforensics_dataset(self, source_dir: str, val_split: float = 0.2):
        """Setup FaceForensics++ dataset with train/val split"""
        print("Setting up FaceForensics++ dataset...")
        
        # Find all real and fake images
        real_images = []
        fake_images = []
        
        # Look for common FaceForensics++ structure
        ff_dirs = [
            Path(source_dir) / "original" / "youtube",
            Path(source_dir) / "manipulated_sequences" / "Deepfakes"
        ]
        
        # Real images from original sequences
        if ff_dirs[0].exists():
            for img_path in ff_dirs[0].rglob("*.jpg"):
                real_images.append(img_path)
        
        # Fake images from manipulated sequences
        if ff_dirs[1].exists():
            for img_path in ff_dirs[1].rglob("*.jpg"):
                fake_images.append(img_path)
        
        print(f"Found {len(real_images)} real images")
        print(f"Found {len(fake_images)} fake images")
        
        # Split datasets
        self._split_and_copy(real_images, fake_images, val_split)
    
    def setup_tiny_genimage_dataset(self, source_dir: str, val_split: float = 0.2):
        """Setup Tiny GenImage dataset"""
        print("Setting up Tiny GenImage dataset...")
        
        source_path = Path(source_dir)
        real_images = []
        fake_images = []
        
        # Look for real and fake directories
        real_dir = source_path / "real"
        fake_dir = source_path / "fake"
        
        if real_dir.exists():
            real_images = list(real_dir.glob("*.jpg"))
        if fake_dir.exists():
            fake_images = list(fake_dir.glob("*.jpg"))
        
        print(f"Found {len(real_images)} real images")
        print(f"Found {len(fake_images)} fake images")
        
        self._split_and_copy(real_images, fake_images, val_split)
    
    def _split_and_copy(self, real_images: List[Path], fake_images: List[Path], val_split: float):
        """Split and copy images to train/val directories"""
        random.seed(42)  # For reproducible splits
        
        # Split real images
        val_real_count = int(len(real_images) * val_split)
        random.shuffle(real_images)
        val_real = real_images[:val_real_count]
        train_real = real_images[val_real_count:]
        
        # Split fake images
        val_fake_count = int(len(fake_images) * val_split)
        random.shuffle(fake_images)
        val_fake = fake_images[:val_fake_count]
        train_fake = fake_images[val_fake_count:]
        
        # Copy files
        print("Copying files to organized structure...")
        
        # Copy training images
        for img_path in train_real:
            shutil.copy2(img_path, self.base_dir / "train" / "real" / img_path.name)
        
        for img_path in train_fake:
            shutil.copy2(img_path, self.base_dir / "train" / "fake" / img_path.name)
        
        # Copy validation images
        for img_path in val_real:
            shutil.copy2(img_path, self.base_dir / "val" / "real" / img_path.name)
        
        for img_path in val_fake:
            shutil.copy2(img_path, self.base_dir / "val" / "fake" / img_path.name)
        
        print(f"Dataset organized:")
        print(f"Train: {len(train_real)} real, {len(train_fake)} fake")
        print(f"Val: {len(val_real)} real, {len(val_fake)} fake")
    
    def create_test_dataset(self, test_dir: str = "test-images"):
        """Create a small test dataset for quick testing"""
        print("Creating test dataset...")
        
        test_path = Path(test_dir)
        test_path.mkdir(exist_ok=True)
        
        # Create subdirectories
        (test_path / "real").mkdir(exist_ok=True)
        (test_path / "fake").mkdir(exist_ok=True)
        
        # Copy a few samples from training set
        train_real = list((self.base_dir / "train" / "real").glob("*.jpg"))
        train_fake = list((self.base_dir / "train" / "fake").glob("*.jpg"))
        
        # Copy 2 images from each class for testing
        for i, img_path in enumerate(train_real[:2]):
            shutil.copy2(img_path, test_path / "real" / f"real{i+1}.jpg")
        
        for i, img_path in enumerate(train_fake[:2]):
            shutil.copy2(img_path, test_path / "fake" / f"fake{i+1}.jpg")
        
        print(f"Test dataset created in {test_dir}")
    
    def get_dataset_stats(self) -> dict:
        """Get dataset statistics"""
        stats = {}
        
        for split in ["train", "val"]:
            for label in ["real", "fake"]:
                path = self.base_dir / split / label
                count = len(list(path.glob("*.jpg")))
                stats[f"{split}_{label}"] = count
        
        return stats

def main():
    """Main setup function"""
    print("Deepfake Detection Dataset Setup")
    print("=" * 50)
    
    setup = DatasetSetup()
    
    print("\nDataset Setup Options:")
    print("1. Download datasets from Kaggle")
    print("2. Setup from existing FaceForensics++ dataset")
    print("3. Setup from existing Tiny GenImage dataset")
    print("4. Create test dataset from existing data")
    print("5. Show dataset statistics")
    
    choice = input("\nSelect option (1-5): ").strip()
    
    if choice == "1":
        datasets = setup.download_kaggle_datasets()
        if datasets:
            print("Datasets downloaded successfully!")
            print(f"FaceForensics++: {datasets['faceforensics']}")
            print(f"Tiny GenImage: {datasets['genimage']}")
            
            # Ask if user wants to organize them
            organize = input("Organize downloaded datasets? (y/n): ").strip().lower()
            if organize == 'y':
                if os.path.exists(datasets['faceforensics']):
                    setup.setup_faceforensics_dataset(datasets['faceforensics'])
                if os.path.exists(datasets['genimage']):
                    setup.setup_tiny_genimage_dataset(datasets['genimage'])
    
    elif choice == "2":
        source_dir = input("Enter FaceForensics++ dataset path: ").strip()
        if os.path.exists(source_dir):
            setup.setup_faceforensics_dataset(source_dir)
        else:
            print("Path does not exist")
    
    elif choice == "3":
        source_dir = input("Enter Tiny GenImage dataset path: ").strip()
        if os.path.exists(source_dir):
            setup.setup_tiny_genimage_dataset(source_dir)
        else:
            print("Path does not exist")
    
    elif choice == "4":
        setup.create_test_dataset()
    
    elif choice == "5":
        stats = setup.get_dataset_stats()
        print("\nDataset Statistics:")
        for key, value in stats.items():
            print(f"{key}: {value}")
    
    else:
        print("Invalid choice")

if __name__ == "__main__":
    main()
