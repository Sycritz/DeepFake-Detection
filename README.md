# Deepfake Detection System

A robust deepfake detection system using dual-stream neural networks with spatial and frequency analysis.

## Overview

This project implements a state-of-the-art deepfake detection model that combines spatial features from Swin Transformer with frequency domain analysis using EfficientNet. The system is designed for real-world deployment with optimized training pipelines and comprehensive evaluation metrics.

## Architecture

### Dual-Stream Network Design

The model employs a dual-stream architecture that processes images through two parallel pathways:

1. **Spatial Stream**: Swin Transformer backbone for hierarchical spatial feature extraction
2. **Frequency Stream**: EfficientNet backbone for frequency domain analysis (DCT/FFT)

### Key Components

- **Swin Transformer**: Extracts multi-scale spatial features with shifted window attention
- **Frequency Encoder**: Processes DCT/FFT coefficients to capture compression artifacts
- **Cross-Attention Mechanism**: Fuses spatial and frequency features through attention-based interaction
- **Patch-Level Fusion**: Combines features at patch level for fine-grained analysis

### Model Specifications

- **Parameters**: 32.6M trainable parameters
- **Input Size**: 224×224 RGB images
- **Frequency Types**: DCT (primary), FFT (alternative)
- **Output**: Binary classification (real/fake)

## Training Methodology

### Dataset Preparation

The system supports multiple dataset formats:
- **FaceForensics++**: High-quality manipulated videos
- **Tiny GenImage**: AI-generated images
- **Custom datasets**: Flexible data loading pipeline

### Training Pipeline

1. **Data Balancing**: Automatic class balancing to prevent bias (15,828 samples per class)
2. **Mixed Precision Training**: Optimized for T4 GPU with automatic mixed precision
3. **Gradient Accumulation**: Effective batch size of 128 for stable training
4. **Checkpointing**: Robust resume capability with best model tracking
5. **F1 Score Optimization**: Primary metric for model selection

### Training Configuration

- **Epochs**: 50 (typical convergence)
- **Batch Size**: 64 (32 per GPU)
- **Learning Rate**: 2e-4 with cosine annealing
- **Optimizer**: AdamW with weight decay 1e-4
- **Frequency Type**: DCT for compression artifact detection

## Performance Metrics

The system evaluates models using comprehensive metrics:

- **Accuracy**: Overall classification accuracy
- **Precision/Recall**: Class-wise performance
- **F1 Score**: Weighted F1 for class balance
- **Confusion Matrix**: TP/FP/TN/FN analysis
- **Inference Speed**: Real-time performance metrics

## Installation and Setup

### Prerequisites

```bash
pip install torch torchvision timm opencv-python scikit-learn
pip install webdataset  # Optional: for large dataset streaming
```

### Dataset Setup

```bash
python setup_dataset.py
# Follow interactive prompts to organize your dataset
```

### Model Testing

```bash
python test_model.py --model models/best_model.pth --data datasets/val
# Quick test with auto-detection:
python test_model.py
```

## Project Structure

```
DeepFake-Detection/
├── models/                 # Trained model checkpoints
│   └── best_model.pth     # Best performing model
├── datasets/              # Organized training data
│   ├── train/            # Training split
│   └── val/              # Validation split
├── test-images/          # Quick test samples
├── setup_dataset.py      # Dataset preparation script
├── test_model.py         # Model evaluation script
└── README.md            # This file
```

## Training Results

The model achieves strong performance on balanced datasets:

- **Validation Accuracy**: ~94-96%
- **F1 Score**: ~0.95 (weighted)
- **Inference Speed**: ~15-20 FPS on T4 GPU
- **Memory Usage**: ~6GB GPU memory

## Model Improvements

### Performance Enhancements

1. **Frequency Analysis**: DCT coefficients capture compression artifacts missed by spatial-only models
2. **Cross-Attention**: Effective fusion of multi-modal features
3. **Class Balancing**: Prevents overfitting to majority class
4. **Mixed Precision**: Faster training without accuracy loss

### Technical Optimizations

- **WebDataset Support**: Efficient streaming for large datasets
- **Gradient Checkpointing**: Memory-efficient training
- **Learning Rate Scheduling**: Cosine annealing for stable convergence
- **Robust Checkpointing**: Resume training from any point

## Deployment Considerations

### Production Deployment

- **Model Size**: 348MB checkpoint file
- **GPU Requirements**: Minimum 6GB VRAM (T4 or better)
- **CPU Inference**: Supported with reduced performance
- **Batch Processing**: Optimized for throughput

### Real-time Applications

- **Video Processing**: Frame-by-frame analysis
- **Image Verification**: Single image classification
- **Batch Analysis**: Large-scale dataset evaluation

## Technical Implementation Details

### Frequency Feature Extraction

The system uses Discrete Cosine Transform (DCT) to capture frequency domain artifacts:

```python
def extract_frequency_features(image_np, freq_type="dct"):
    if freq_type == "dct":
        ycbcr = cv2.cvtColor(image_np, cv2.COLOR_RGB2YCrCb)
        dct = cv2.dct(ycbcr[:, :, 0].astype(np.float32))
        freq_features = np.log1p(np.abs(dct))
```

### Cross-Attention Fusion

Spatial and frequency features are fused through cross-attention:

```python
class CrossAttention(nn.Module):
    def forward(self, query, key, value):
        # Multi-head attention between spatial and frequency features
        attn = (Q @ K.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        return (attn @ V)
```

## Evaluation Protocol

### Standard Testing

Use the provided test script for comprehensive evaluation:

```bash
python test_model.py --model models/best_model.pth --data datasets/val --batch-size 32
```

### Custom Evaluation

The test script supports:
- Custom dataset paths
- Different batch sizes
- DCT vs FFT frequency analysis
- JSON result export

## Acknowledgments

This implementation builds upon research in:
- Vision Transformers for deepfake detection
- Frequency domain analysis in image forensics
- Cross-modal attention mechanisms

## License

This project is provided for research and educational purposes.
