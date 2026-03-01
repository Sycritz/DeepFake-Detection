# Deepfake Detection Dataset Summary

## Dataset Structure
```
datasets/
├── train/
│   ├── real/     (15,828 images)
│   └── fake/     (20,000 images)
└── val/
    ├── real/     (2,000 images)
    └── fake/     (2,000 images)
```

## Dataset Composition

### Training Dataset (30,000 images total)
- **Real images (15,828)**:
  - 10,000 from FaceForensics C23 (extracted faces)
  - 5,828 from Tiny GenImage Nature dataset
- **Fake images (20,000)**:
  - 10,000 from FaceForensics C23 (DeepFakeDetection, Deepfakes, Face2Face, FaceShifter, FaceSwap, NeuralTextures)
  - 10,000 from Tiny GenImage diffusion models (prioritizing Stable Diffusion v1.5, Midjourney, ADM, BigGAN, VQDM, glide, wukong)

### Validation Dataset (4,000 images total)
- **Real images (2,000)**:
  - 1,000 from FaceForensics C23 validation set
  - 1,000 from Tiny GenImage Nature dataset
- **Fake images (2,000)**:
  - 1,000 from FaceForensics C23 validation set
  - 1,000 from Tiny GenImage diffusion models

## Key Features
- **Balanced dataset**: Good mix of real and fake images
- **Diverse fake sources**: Includes multiple manipulation techniques
- **Prioritized diffusion models**: Focus on modern AI-generated content
- **Proper train/val split**: Suitable for model training and evaluation
- **Consistent image format**: All images are in JPEG/JPG format
- **Organized structure**: Clear directory layout for easy loading

## Usage
This dataset is ready for deepfake detection model training. The images are organized in a standard classification format that can be easily loaded with PyTorch DataLoader or TensorFlow datasets.

## File Naming Convention
- FaceForensics images: `ff_[original_filename]`
- Tiny GenImage images: `tg_[original_filename]`

This helps track the source of each image during analysis.
