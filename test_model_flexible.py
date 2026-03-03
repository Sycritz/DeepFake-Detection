#!/usr/bin/env python3
"""
Simple Model Testing Script - Adapts to different model architectures
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

def test_model_flexible(model_path: str, data_dir: str, batch_size: int = 32, freq_type: str = "dct"):
    """Test model with flexible architecture detection"""
    
    print("Flexible Model Testing")
    print("=" * 50)
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model and detect architecture
    print(f"Loading model from: {model_path}")
    checkpoint = torch.load(model_path, map_location="cpu")
    
    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint
    
    # Detect model type from architecture
    model = None
    
    # Check for FinalPrototype architecture (newer)
    if any("freq_cnn" in key for key in state_dict.keys()):
        print("Detected: FinalPrototype architecture")
        # This is a different architecture - need to create it
        import timm
        
        class FinalPrototypeModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.swin_backbone = timm.create_model(
                    "swin_tiny_patch4_window7_224",
                    pretrained=False,
                    features_only=True,
                    out_indices=(3,),
                )
                spatial_dim = self.swin_backbone.feature_info.channels()[-1]
                
                # Frequency CNN from the actual model
                self.freq_cnn = nn.Sequential(
                    nn.Conv2d(3, 64, 3, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(64, 128, 3, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(128, 256, 3, padding=1),
                    nn.ReLU(),
                    nn.AdaptiveAvgPool2d((1, 1)),
                    nn.Flatten(),
                    nn.Linear(256, 256)
                )
                
                # Fusion layer with correct dimensions
                self.fusion = nn.Sequential(
                    nn.Linear(spatial_dim + 256, 512),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(512, 2),
                )

            def forward(self, spatial_input, freq_input):
                # Spatial features
                feat = self.swin_backbone(spatial_input)[0]  # [B, H, W, C]
                B, H, W, C = feat.shape
                spatial_flat = feat.view(B, H * W, C)
                
                # Frequency features
                freq_features = self.freq_cnn(freq_input)  # [B, 256]
                
                # Average spatial features across patches
                spatial_avg = spatial_flat.mean(dim=1)  # [B, C]
                
                # Concatenate and classify
                combined = torch.cat([spatial_avg, freq_features], dim=1)
                output = self.fusion(combined)
                
                return output
        
        model = FinalPrototypeModel().to(device)
        model.load_state_dict(state_dict, strict=False)
        print("FinalPrototype model loaded successfully")
    
    else:
        print("Detected: Standard architecture")
        # Try to load as standard model
        try:
            # Import the standard model architecture
            import sys
            sys.path.append('.')
            from test_model import ImprovedDeepfakeDetector
            model = ImprovedDeepfakeDetector().to(device)
            model.load_state_dict(state_dict, strict=False)
            print("Standard model loaded successfully")
        except Exception as e:
            print(f"Could not load standard model: {e}")
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
                print(f"   Processed batch {batch_idx + 1}/{len(dataloader)}")
    
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
    print(f"   True Positives:  {tp}")
    print(f"   True Negatives:  {tn}")
    print(f"   False Positives: {fp}")
    print(f"   False Negatives: {fn}")
    print()
    print("Performance Metrics:")
    print(f"   Avg Inference Time: {avg_inference_time*1000:.2f}ms/batch")
    print(f"   Total Inference Time: {total_inference_time:.2f}s")
    print(f"   Throughput: {throughput:.2f} samples/sec")
    
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
    results_path = "test_results_flexible.json"
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
    
    test_model_flexible(args.model, args.data, args.batch_size, args.freq_type)

if __name__ == "__main__":
    # Quick test usage
    if len(os.sys.argv) == 1:
        print("Quick Test Mode")
        print("Usage examples:")
        print("python test_model_flexible.py --model models/Prototype.pth --data test-images")
        print("python test_model_flexible.py --model models/FinalPrototye.pth --data test-images --batch-size 16")
        print()
        
        # Try to find model and test data automatically
        model_path = "models/Prototype.pth"
        data_path = "test-images"
        
        if os.path.exists(model_path) and os.path.exists(data_path):
            print(f"Auto-detected: {model_path} and {data_path}")
            test_model_flexible(model_path, data_path)
        else:
            print("Could not auto-detect model and test data")
            print("Please specify --model and --data arguments")
    else:
        main()
