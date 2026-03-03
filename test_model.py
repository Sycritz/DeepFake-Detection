#!/usr/bin/env python3
"""
Deepfake Detection Model Testing Script
Quick and simple model evaluation with TP, FP, F1 metrics
"""

import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score, recall_score
import cv2
import json
import argparse

# Import model architecture
import timm

class CrossAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8, qkv_bias: bool = False):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        B, N, C = query.shape

        Q = self.q_proj(query).reshape(B, N, self.num_heads, C // self.num_heads).transpose(1, 2)
        K = self.k_proj(key).reshape(B, N, self.num_heads, C // self.num_heads).transpose(1, 2)
        V = self.v_proj(value).reshape(B, N, self.num_heads, C // self.num_heads).transpose(1, 2)

        attn = (Q @ K.transpose(-2, -1)) * self.scale
        attn = nn.functional.softmax(attn, dim=-1)

        out = (attn @ V).transpose(1, 2).reshape(B, N, C)
        out = self.out_proj(out)
        return out

class FrequencyEncoder(nn.Module):
    def __init__(self, feature_dim: int = 256):
        super().__init__()
        self.backbone = timm.create_model("efficientnet_b0", pretrained=False, num_classes=0)
        self.feature_dim = feature_dim
        self.proj = nn.Linear(self.backbone.num_features, feature_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        return self.proj(features)

class PatchLevelFusion(nn.Module):
    def __init__(self, spatial_dim: int, freq_dim: int = 256, num_heads: int = 8):
        super().__init__()
        self.spatial_dim = spatial_dim
        self.freq_dim = freq_dim

        self.spatial_proj = nn.Linear(spatial_dim, freq_dim)
        self.freq_proj = nn.Linear(freq_dim, freq_dim)

        self.cross_attention = CrossAttention(dim=freq_dim, num_heads=num_heads)

        self.fusion = nn.Sequential(
            nn.Linear(freq_dim * 2, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 2),
        )

    def forward(self, spatial_features: torch.Tensor, freq_features: torch.Tensor) -> torch.Tensor:
        B, H, W, C = spatial_features.shape
        spatial_flat = spatial_features.view(B, H * W, C)
        
        spatial_proj = self.spatial_proj(spatial_flat)
        freq_proj = self.freq_proj(freq_features).unsqueeze(1).expand(-1, H * W, -1)

        attended = self.cross_attention(spatial_proj, freq_proj, freq_proj)
        combined = torch.cat([attended, freq_proj], dim=-1)
        pooled = combined.mean(dim=1)

        return self.fusion(pooled)

class ImprovedDeepfakeDetector(nn.Module):
    def __init__(self):
        super().__init__()
        self.swin_backbone = timm.create_model(
            "swin_tiny_patch4_window7_224",
            pretrained=False,
            features_only=True,
            out_indices=(3,),
        )
        spatial_dim = self.swin_backbone.feature_info.channels()[-1]

        self.freq_encoder = FrequencyEncoder(feature_dim=256)
        self.fusion = PatchLevelFusion(spatial_dim=spatial_dim, freq_dim=256)

    def forward(self, spatial_input: torch.Tensor, freq_input: torch.Tensor) -> torch.Tensor:
        feat = self.swin_backbone(spatial_input)[0]
        freq_features = self.freq_encoder(freq_input)
        return self.fusion(feat, freq_features)

def extract_frequency_features(image_np: np.ndarray, freq_type: str = "dct") -> np.ndarray:
    if freq_type == "dct":
        ycbcr = cv2.cvtColor(image_np, cv2.COLOR_RGB2YCrCb)
        dct = cv2.dct(ycbcr[:, :, 0].astype(np.float32))
        freq_features = np.log1p(np.abs(dct))
    elif freq_type == "fft":
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
        fft = np.fft.fft2(gray)
        magnitude = np.abs(np.fft.fftshift(fft))
        freq_features = np.log1p(magnitude)
    else:
        raise ValueError(f"Unsupported freq_type: {freq_type}")

    freq_min, freq_max = float(freq_features.min()), float(freq_features.max())
    if freq_max > freq_min:
        freq_features = (freq_features - freq_min) / (freq_max - freq_min)

    return freq_features

def frequency_tensor_from_pil(image: Image.Image, size: tuple = (224, 224), freq_type: str = "dct") -> torch.Tensor:
    image_np = np.array(image.resize(size))
    freq_features = extract_frequency_features(image_np, freq_type)
    freq_tensor = torch.from_numpy(freq_features).float()
    if freq_tensor.dim() == 2:
        freq_tensor = freq_tensor.unsqueeze(0).repeat(3, 1, 1)
    return freq_tensor

class TestDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir: str, transform=None, freq_type: str = "dct"):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.freq_type = freq_type
        self.samples = []
        self._load_samples()

    def _load_samples(self):
        for label, class_name in enumerate(["real", "fake"]):
            class_dir = self.data_dir / class_name
            if not class_dir.exists():
                continue
            for img_path in class_dir.glob("*.jpg"):
                self.samples.append((img_path, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")

        spatial_tensor = self.transform(image) if self.transform else transforms.ToTensor()(image)
        freq_tensor = frequency_tensor_from_pil(image, size=(224, 224), freq_type=self.freq_type)
        return spatial_tensor, freq_tensor, int(label)

def test_model(model_path: str, data_dir: str, batch_size: int = 32, freq_type: str = "dct"):
    """Test model performance with comprehensive metrics"""
    
    print("Deepfake Detection Model Testing")
    print("=" * 50)
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model
    print(f"Loading model from: {model_path}")
    model = ImprovedDeepfakeDetector().to(device)
    
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location="cpu")
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)
        print("Model loaded successfully")
    else:
        print(f"Model file not found: {model_path}")
        return
    
    # Setup data transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # Load dataset
    print(f"Loading test data from: {data_dir}")
    dataset = TestDataset(data_dir, transform=transform, freq_type=freq_type)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    if len(dataset) == 0:
        print("No test data found")
        return
    
    print(f"Test dataset size: {len(dataset)} samples")
    
    # Test model
    model.eval()
    all_preds = []
    all_labels = []
    inference_times = []
    
    print("\nRunning inference...")
    
    with torch.no_grad():
        for batch_idx, (spatial_imgs, freq_imgs, labels) in enumerate(dataloader):
            spatial_imgs = spatial_imgs.to(device)
            freq_imgs = freq_imgs.to(device)
            
            # Measure inference time
            start_time = time.time()
            
            outputs = model(spatial_imgs, freq_imgs)
            preds = outputs.argmax(dim=1)
            
            inference_time = time.time() - start_time
            inference_times.append(inference_time)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            
            if batch_idx % 10 == 0:
                print(f"Processed batch {batch_idx + 1}/{len(dataloader)}")
    
    # Calculate metrics
    print("\nComputing metrics...")
    
    # Basic metrics
    accuracy = (np.array(all_preds) == np.array(all_labels)).mean()
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    f1 = f1_score(all_labels, all_preds, average='weighted')
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    tn, fp, fn, tp = cm.ravel()
    
    # Performance metrics
    avg_inference_time = np.mean(inference_times)
    total_inference_time = sum(inference_times)
    throughput = len(all_preds) / total_inference_time
    
    # Print results
    print("\nModel Performance Results:")
    print("=" * 40)
    print(f"Accuracy:      {accuracy:.4f}")
    print(f"Precision:     {precision:.4f}")
    print(f"Recall:        {recall:.4f}")
    print(f"F1 Score:      {f1:.4f}")
    print()
    print("Confusion Matrix:")
    print(f"True Positives:  {tp}")
    print(f"True Negatives:  {tn}")
    print(f"False Positives: {fp}")
    print(f"False Negatives: {fn}")
    print()
    print("Performance Metrics:")
    print(f"Avg Inference Time: {avg_inference_time*1000:.2f}ms/batch")
    print(f"Total Inference Time: {total_inference_time:.2f}s")
    print(f"Throughput: {throughput:.2f} samples/sec")
    
    # Detailed classification report
    print("\nDetailed Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=["Real", "Fake"]))
    
    # Save results
    results = {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "confusion_matrix": {
            "true_positives": int(tp),
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn)
        },
        "performance": {
            "avg_inference_time_ms": float(avg_inference_time * 1000),
            "total_inference_time_s": float(total_inference_time),
            "throughput_samples_per_sec": float(throughput)
        }
    }
    
    # Save to file
    results_path = "test_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {results_path}")
    print("Testing completed!")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Test Deepfake Detection Model")
    parser.add_argument("--model", required=True, help="Path to model checkpoint")
    parser.add_argument("--data", required=True, help="Path to test dataset")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--freq-type", default="dct", choices=["dct", "fft"], help="Frequency type")
    
    args = parser.parse_args()
    
    test_model(args.model, args.data, args.batch_size, args.freq_type)

if __name__ == "__main__":
    # Quick test usage
    if len(os.sys.argv) == 1:
        print("Quick Test Mode")
        print("Usage examples:")
        print("python test_model.py --model models/best_model.pth --data datasets/val")
        print("python test_model.py --model models/best_model.pth --data test-images --batch-size 16")
        print()
        
        # Try to find model and test data automatically
        model_path = "models/best_model.pth"
        data_path = "test-images"
        
        if os.path.exists(model_path) and os.path.exists(data_path):
            print(f"Auto-detected: {model_path} and {data_path}")
            test_model(model_path, data_path)
        else:
            print("Could not auto-detect model and test data")
            print("Please specify --model and --data arguments")
    else:
        main()
