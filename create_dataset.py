#!/usr/bin/env python3
"""
Deepfake Dataset Creation Script
Creates training and validation datasets for deepfake detection model.

Training dataset:
- 20,000 images from FaceForensics C23 (10K real, 10K fake)
- 20,000 images from Tiny GenImage (10K real, 10K fake, prioritizing diffusion models)

Validation dataset: Smaller and versatile
"""

import os
import shutil
import random
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
import argparse

def setup_directories(base_dir: str = "datasets") -> Dict[str, Path]:
    """Create the necessary directory structure."""
    base_path = Path(base_dir)
    
    directories = {
        'train_real': base_path / "train" / "real",
        'train_fake': base_path / "train" / "fake", 
        'val_real': base_path / "val" / "real",
        'val_fake': base_path / "val" / "fake"
    }
    
    for dir_path in directories.values():
        dir_path.mkdir(parents=True, exist_ok=True)
        print(f"Created directory: {dir_path}")
    
    return directories

def get_faceforensics_images(base_path: str, split: str = "train") -> Tuple[List[str], List[str]]:
    """Get real and fake image paths from FaceForensics C23 dataset."""
    base_path = Path(base_path) / split
    
    real_images = []
    fake_images = []
    
    # Real images
    real_dir = base_path / "Real"
    if real_dir.exists():
        real_images = [str(img) for img in real_dir.glob("*.jpg")]
        print(f"Found {len(real_images)} real images in FaceForensics {split}")
    
    # Fake images from all subdirectories
    fake_dirs = ["DeepFakeDetection", "Deepfakes", "Face2Face", "FaceShifter", "FaceSwap", "NeuralTextures"]
    for fake_dir in fake_dirs:
        fake_path = base_path / fake_dir
        if fake_path.exists():
            fake_images.extend([str(img) for img in fake_path.glob("*.jpg")])
    
    print(f"Found {len(fake_images)} fake images in FaceForensics {split}")
    return real_images, fake_images

def get_tiny_genimage_images(base_path: str) -> Tuple[List[str], List[str]]:
    """Get real and fake image paths from Tiny GenImage dataset."""
    base_path = Path(base_path)
    
    real_images = []
    fake_images = []
    
    # Real images
    real_dir = base_path / "Nature"
    if real_dir.exists():
        real_images = [str(img) for img in real_dir.glob("*.JPEG")] + [str(img) for img in real_dir.glob("*.jpg")]
        print(f"Found {len(real_images)} real images in Tiny GenImage")
    
    # Fake images - prioritize diffusion models
    diffusion_dirs = [
        "stable_diffusion_v_1_5", "Midjourney", "ADM", "BigGAN", 
        "VQDM", "glide", "wukong"
    ]
    
    # First collect from diffusion models
    for diffusion_dir in diffusion_dirs:
        diffusion_path = base_path / diffusion_dir
        if diffusion_path.exists():
            fake_images.extend([str(img) for img in diffusion_path.glob("*.JPEG")] + [str(img) for img in diffusion_path.glob("*.jpg")])
    
    print(f"Found {len(fake_images)} fake images from diffusion models in Tiny GenImage")
    return real_images, fake_images

def copy_images(image_paths: List[str], target_dir: Path, prefix: str = "", max_images: int = None) -> int:
    """Copy images to target directory with optional prefix and limit."""
    if max_images:
        image_paths = random.sample(image_paths, min(max_images, len(image_paths)))
    
    copied_count = 0
    for img_path in image_paths:
        try:
            src_path = Path(img_path)
            dst_filename = f"{prefix}_{src_path.name}" if prefix else src_path.name
            dst_path = target_dir / dst_filename
            
            # Avoid filename conflicts
            counter = 1
            original_dst = dst_path
            while dst_path.exists():
                stem = original_dst.stem
                suffix = original_dst.suffix
                dst_path = target_dir / f"{stem}_{counter}{suffix}"
                counter += 1
            
            shutil.copy2(src_path, dst_path)
            copied_count += 1
            
            if copied_count % 1000 == 0:
                print(f"Copied {copied_count} images to {target_dir}")
                
        except Exception as e:
            print(f"Error copying {img_path}: {e}")
    
    print(f"Successfully copied {copied_count} images to {target_dir}")
    return copied_count

def create_validation_dataset(faceforensics_val_path: str, tiny_genimage_path: str, 
                            directories: Dict[str, Path], val_size_per_source: int = 1000):
    """Create a smaller, versatile validation dataset."""
    print("\n=== Creating Validation Dataset ===")
    
    # Get validation images from FaceForensics
    ff_real, ff_fake = get_faceforensics_images(faceforensics_val_path, split="val")
    
    # Get additional images from Tiny GenImage for validation
    tg_real, tg_fake = get_tiny_genimage_images(tiny_genimage_path)
    
    # Sample images for validation
    val_real_ff = random.sample(ff_real, min(val_size_per_source // 2, len(ff_real)))
    val_fake_ff = random.sample(ff_fake, min(val_size_per_source // 2, len(ff_fake)))
    
    val_real_tg = random.sample(tg_real, min(val_size_per_source // 2, len(tg_real)))
    val_fake_tg = random.sample(tg_fake, min(val_size_per_source // 2, len(tg_fake)))
    
    # Copy validation images
    copy_images(val_real_ff, directories['val_real'], prefix="ff", max_images=None)
    copy_images(val_real_tg, directories['val_real'], prefix="tg", max_images=None)
    copy_images(val_fake_ff, directories['val_fake'], prefix="ff", max_images=None)
    copy_images(val_fake_tg, directories['val_fake'], prefix="tg", max_images=None)
    
    total_val = len(val_real_ff) + len(val_real_tg) + len(val_fake_ff) + len(val_fake_tg)
    print(f"Validation dataset created with {total_val} images")

def main():
    parser = argparse.ArgumentParser(description="Create deepfake detection dataset")
    parser.add_argument("--faceforensics_path", type=str, 
                       default="/home/sycrits/.cache/kagglehub/datasets/gradientvoyager/faceforensics-c23-extracted-faces-100k/versions/1/dataset_processed_split",
                       help="Path to FaceForensics C23 dataset")
    parser.add_argument("--tiny_genimage_path", type=str,
                       default="/home/sycrits/.cache/kagglehub/datasets/cartografia/unbiased-tiny-genimage/versions/1",
                       help="Path to Tiny GenImage dataset")
    parser.add_argument("--output_dir", type=str, default="datasets",
                       help="Output directory for datasets")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--val_size", type=int, default=2000, help="Validation set size per source")
    
    args = parser.parse_args()
    
    # Set random seed
    random.seed(args.seed)
    
    print("=== Deepfake Dataset Creation ===")
    print(f"FaceForensics path: {args.faceforensics_path}")
    print(f"Tiny GenImage path: {args.tiny_genimage_path}")
    print(f"Output directory: {args.output_dir}")
    print(f"Random seed: {args.seed}")
    
    # Setup directories
    directories = setup_directories(args.output_dir)
    
    print("\n=== Processing FaceForensics C23 Dataset ===")
    # Get training images from FaceForensics
    ff_real_train, ff_fake_train = get_faceforensics_images(args.faceforensics_path, split="train")
    
    # Copy FaceForensics training images (10K real, 10K fake)
    print("Copying FaceForensics real images...")
    copy_images(ff_real_train, directories['train_real'], prefix="ff", max_images=10000)
    
    print("Copying FaceForensics fake images...")
    copy_images(ff_fake_train, directories['train_fake'], prefix="ff", max_images=10000)
    
    print("\n=== Processing Tiny GenImage Dataset ===")
    # Get training images from Tiny GenImage
    tg_real, tg_fake = get_tiny_genimage_images(args.tiny_genimage_path)
    
    # Copy Tiny GenImage training images (10K real, 10K fake)
    print("Copying Tiny GenImage real images...")
    copy_images(tg_real, directories['train_real'], prefix="tg", max_images=10000)
    
    print("Copying Tiny GenImage fake images (prioritizing diffusion models)...")
    copy_images(tg_fake, directories['train_fake'], prefix="tg", max_images=10000)
    
    # Create validation dataset
    create_validation_dataset(args.faceforensics_path, args.tiny_genimage_path, 
                           directories, args.val_size)
    
    print("\n=== Dataset Creation Complete ===")
    
    # Print final statistics
    print("\nFinal Dataset Statistics:")
    for split in ["train", "val"]:
        for label in ["real", "fake"]:
            dir_path = directories[f'{split}_{label}']
            count = len(list(dir_path.glob("*.jpg")))
            print(f"{split}/{label}: {count} images")
    
    total_train = len(list(directories['train_real'].glob("*.jpg"))) + len(list(directories['train_fake'].glob("*.jpg")))
    total_val = len(list(directories['val_real'].glob("*.jpg"))) + len(list(directories['val_fake'].glob("*.jpg")))
    print(f"\nTotal training images: {total_train}")
    print(f"Total validation images: {total_val}")
    print(f"Total images: {total_train + total_val}")

if __name__ == "__main__":
    main()
