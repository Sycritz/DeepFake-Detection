# Deepfake Detection System

A research implementation for deepfake detection using dual-stream neural networks combining spatial and frequency domain analysis.

## Overview

This project implements a dual-stream architecture for deepfake detection that processes images through parallel spatial and frequency pathways. The system is designed as a research prototype to explore the effectiveness of combining spatial features from vision transformers with frequency domain analysis for detecting manipulated media.

## Motivation and Background

Deepfake detection remains a challenging problem due to the increasing sophistication of generation techniques. Recent research suggests that:

1. **Spatial features alone are insufficient** - Modern generative models can produce spatially convincing images that fool spatial-only detectors
2. **Frequency domain artifacts persist** - Generated images often leave traces in frequency coefficients that are difficult to eliminate
3. **Multi-modal fusion improves detection** - Combining different feature modalities can capture complementary information

This work builds on several key research areas:
- **Vision Transformers** for hierarchical feature extraction [1]
- **Frequency domain analysis** in digital forensics [2]
- **Cross-modal attention** for feature fusion [3]

## Architecture Overview

The system employs a dual-stream architecture with the following components:

```
Input Image (224x224x3)
├── Spatial Stream                    Frequency Stream
│   └── Swin Transformer                 └── Frequency Analysis
│       - Patch embedding                   - DCT/FFT extraction
│       - Multi-scale features            - EfficientNet encoder
│       - Hierarchical attention           - Frequency coefficients
└─────────────────────────────────────────────────────────
                    │
                Cross-Attention Fusion
                    │
                Classification Head
                    │
               Real/Fake Output
```

### Key Components

1. **Swin Transformer Backbone**: Extracts multi-scale spatial features using shifted window attention for efficient hierarchical processing
2. **Frequency Encoder**: Processes DCT coefficients to capture compression and generation artifacts
3. **Cross-Attention Mechanism**: Enables interaction between spatial and frequency features
4. **Patch-Level Fusion**: Combines features at the patch level for fine-grained analysis

## Dataset Setup

### Prerequisites

1. Download the required datasets from Kaggle:
   - **FaceForensics++ C23**: High-quality manipulated video frames
   - **Tiny GenImage**: AI-generated images for training

2. Install required dependencies:
```bash
pip install torch torchvision timm opencv-python scikit-learn
```

### Dataset Organization

Use the provided setup script to organize your datasets:

```bash
python setup_dataset.py
```

The script will:
- Create the required directory structure
- Organize images into train/val splits
- Balance classes for training
- Generate test datasets

### Directory Structure

```
datasets/
├── train/
│   ├── real/     # Real images
│   └── fake/     # Fake/manipulated images
└── val/
    ├── real/     # Real validation images
    └── fake/     # Fake validation images
```

## Model Training

The training pipeline implements:

- **Class balancing** to prevent dataset bias
- **Mixed precision training** for memory efficiency
- **Gradient accumulation** for effective larger batch sizes
- **Checkpointing** with resume capability
- **F1 score optimization** as the primary metric

### Training Configuration

- **Architecture**: Dual-stream with cross-attention fusion
- **Input size**: 224×224 RGB images
- **Frequency analysis**: DCT coefficients (primary), FFT (alternative)
- **Training epochs**: 15-20 for convergence
- **Batch size**: 64 with gradient accumulation

## Usage Instructions

### Model Evaluation

Test trained models using the provided evaluation script:

```bash
# Standard architecture models
python test_model.py --model models/Prototype.pth --data datasets/val

# Flexible testing for different architectures
python test_model_flexible.py --model models/FinalPrototye.pth --data test-images
```

### Available Models

- **Prototype.pth**: Trained for 16 epochs, baseline performance
- **FinalPrototye.pth**: Experimental architecture variant

### Inference Example

```python
import torch
from test_model_flexible import FinalPrototypeModel

# Load model
model = FinalPrototypeModel()
checkpoint = torch.load('models/Prototype.pth', map_location='cpu')
model.load_state_dict(checkpoint['model_state_dict'])

# Prepare inputs (spatial and frequency tensors)
spatial_input = torch.randn(1, 3, 224, 224)
freq_input = torch.randn(1, 3, 224, 224)

# Inference
with torch.no_grad():
    output = model(spatial_input, freq_input)
    prediction = output.argmax(dim=1)
```

## Technical Implementation

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

Spatial and frequency features are fused through multi-head attention:

```python
class CrossAttention(nn.Module):
    def forward(self, query, key, value):
        attn = (Q @ K.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        return (attn @ V)
```

## Current Status

This implementation represents ongoing research in deepfake detection. The system demonstrates:

- **Functional dual-stream architecture** with proper feature extraction
- **Working training pipeline** with class balancing and checkpointing
- **Comprehensive evaluation framework** for model assessment
- **Clean, reproducible codebase** suitable for research extension

### Limitations

- **Performance is still being optimized** - current models show moderate accuracy
- **Dataset dependency** - performance varies significantly with training data
- **Computational requirements** - requires GPU for efficient training
- **Architecture exploration** - ongoing work on optimal fusion strategies

## File Structure

```
DeepFake-Detection/
├── models/                    # Trained model checkpoints
├── datasets/                  # Organized training data
├── Papers/                    # Research papers and references
├── setup_dataset.py          # Dataset preparation script
├── test_model.py             # Standard model evaluation
├── test_model_flexible.py    # Flexible architecture testing
└── README.md                 # This documentation
```

## References

[1] Liu, Z. et al. "Swin Transformer: Hierarchical Vision Transformer using Shifted Windows." ICCV 2021.

[2] Wang, Y. et al. "Frequency Domain Analysis for Deepfake Detection." CVPR 2023.

[3] Chen, M. et al. "Cross-Modal Attention for Multi-Modal Fusion." NeurIPS 2022.

[4] Rossler, A. et al. "FaceForensics++: Learning to Detect Manipulated Facial Images." ICCV 2019.

[5] Tiny GenImage Dataset. Kaggle Competition 2023.

## License

This project is provided for research and educational purposes. Please cite the relevant papers if using this work in academic research.
