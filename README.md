# DeepFake Detection

A comprehensive deepfake detection system that combines spatial and frequency domain analysis using hybrid transformer-CNN architectures.

## Project Overview

This project implements a robust deepfake detection pipeline that leverages both spatial features from Swin Transformer and frequency domain features from custom CNN architectures. The system is designed to detect AI-generated images and manipulated media with high accuracy across multiple generation techniques.

## Architecture

### Spatial Feature Extraction

- **Swin-T Backbone**: Hierarchical Vision Transformer using Shifted Windows
- **Pre-trained**: ImageNet weights for transfer learning
- **Output**: 768-dimensional spatial feature vectors

### Frequency Domain Analysis

- **DCT Processing**: Discrete Cosine Transform for compression artifact detection
- **FFT Analysis**: Fast Fourier Transform for frequency pattern analysis
- **Custom CNN**: 3-layer convolutional network for frequency feature extraction

### Feature Fusion

- **Multi-modal Fusion**: Combines spatial and frequency features
- **Classification Head**: Binary classification (Real vs Fake)

## Research Inspiration

This implementation draws inspiration from several key research papers:

### Spatial Feature Extraction

- **"Swin Transformer: Hierarchical Vision Transformer using Shifted Windows"** (Liu et al., 2021)
  - Hierarchical architecture with shifted windowing mechanism
  - Efficient attention computation for vision tasks

### Frequency Domain Analysis

- **"Frequency Domain Analysis for Deepfake Detection"** (Wang et al., 2020)
  - DCT coefficient analysis for detecting compression artifacts
  - Frequency domain feature extraction for forgery detection

### Hybrid Architectures

- **"Hybrid CNN-Transformer Models for Image Classification"** (Chen et al., 2022)
  - Combining convolutional and transformer features
  - Multi-modal feature fusion strategies

### Deepfake Detection

- **"FaceForensics++: Learning to Detect Manipulated Facial Images"** (Rössler et al., 2019)
  - Comprehensive dataset for deepfake detection
  - Multi-category manipulation detection

## Dataset

### Training Data

- **FaceForensics C23**: 20,000 images (10,000 real, 10,000 fake)
  - Multiple manipulation techniques: DeepFakeDetection, Deepfakes, Face2Face, FaceShifter, FaceSwap, NeuralTextures
- **Tiny GenImage**: 15,828 images (5,828 real, 10,000 fake)
  - Real images from Nature dataset
  - AI-generated images from diffusion models: Stable Diffusion, Midjourney, ADM, BigGAN, VQDM, glide, wukong

### Validation Data

- **4,000 images**: Balanced real/fake split
- **Cross-source**: Mixed from both datasets for generalization

## Project Structure

```
DeepFake-Detection/
├── datasets/                    # Training and validation data
│   ├── train/                   # Training images (35,828 total)
│   │   ├── real/               # Real images
│   │   └── fake/               # Fake images
│   └── val/                    # Validation images (4,000 total)
│       ├── real/               # Real images
│       └── fake/               # Fake images
├── deepfake_data.py            # Data pipeline and loading
├── Swin-T_download.py          # Swin-T model testing
├── create_dataset.py           # Dataset creation script
├── plan.md                     # Implementation roadmap
└── dataset_summary.md          # Dataset statistics
```

## Implementation Status

### Completed Components

- [x] Dataset creation and organization
- [x] Data pipeline with spatial and frequency features
- [x] Swin-T model integration and testing
- [x] Frequency feature extraction (DCT/FFT)
- [x] Data augmentation and preprocessing

### In Progress

- [ ] Hybrid model architecture implementation
- [ ] Training pipeline and optimization
- [ ] Model evaluation and metrics
- [ ] Inference and deployment

### Planned Features

- [ ] Multi-scale frequency analysis
- [ ] Attention mechanisms for feature fusion
- [ ] Cross-dataset evaluation
- [ ] Model optimization for deployment

## Key Features

### Data Processing

- **Dual-domain features**: Spatial (RGB) and frequency (DCT/FFT) analysis
- **Robust augmentation**: Random cropping, color jitter, rotation, blur
- **Efficient caching**: Frequency feature caching for performance
- **Flexible configuration**: Support for different frequency types and image sizes

### Model Design

- **Transfer learning**: Pre-trained Swin-T for spatial features
- **Custom architecture**: Frequency CNN optimized for artifact detection
- **Memory efficient**: Optimized for Colab GPU constraints
- **Scalable**: Modular design for easy extension

## Performance Targets

- **Accuracy**: >95% on validation set
- **AUC-ROC**: >0.98
- **Inference speed**: <50ms per image
- **Model size**: <500MB for deployment

## Dependencies

- PyTorch >= 1.9.0
- torchvision >= 0.10.0
- timm >= 0.6.0
- OpenCV >= 4.5.0
- NumPy >= 1.21.0
- Pillow >= 8.3.0

## Usage

### Dataset Preparation

```bash
python create_dataset.py
```

### Model Testing

```bash
python Swin-T_download.py
```

### Data Pipeline Validation

```python
from deepfake_data import DatasetConfig, DataModule, FrequencyType

config = DatasetConfig(
    root_dir="datasets",
    batch_size=32,
    frequency_type=FrequencyType.DCT
)

data_module = DataModule(config)
data_module.setup("datasets/train", "datasets/val")
```

## Future Directions

### Research Extensions

- **Multi-modal fusion**: Audio-visual deepfake detection
- **Temporal analysis**: Video deepfake detection
- **Explainable AI**: Feature visualization and interpretation
- **Adversarial robustness**: Defense against adversarial attacks

### Applications

- **Content moderation**: Social media platform integration
- **Forensic analysis**: Media authentication tools
- **Education**: Deepfake awareness and detection training

## Contributing

This project follows academic research standards and welcomes contributions that align with the research objectives outlined in the implementation plan.

## License

This project is for research and educational purposes. Please refer to individual dataset licenses for usage restrictions.
