import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

# Test Swin-T output dimensions
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Create a dummy input
dummy_input = torch.randn(1, 3, 224, 224).to(device)
print(f"Input shape: {dummy_input.shape}")

# Load Swin-T
model = timm.create_model('swin_tiny_patch4_window7_224', pretrained=True)
model = model.to(device)
model.eval()

print(f"Swin-T architecture:")
print(model)

# Test forward pass
with torch.no_grad():
    output = model(dummy_input)
    print(f"Original output shape: {output.shape}")

# Test with Identity head
model.head = nn.Identity().to(device)
with torch.no_grad():
    identity_output = model(dummy_input)
    print(f"Identity head output shape: {identity_output.shape}")

# Test if we need to flatten
if identity_output.dim() > 2:
    flattened = identity_output.view(identity_output.size(0), -1)
    print(f"Flattened shape: {flattened.shape}")
else:
    flattened = identity_output
    print(f"No flattening needed: {flattened.shape}")

# Test frequency CNN
class FrequencyCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, 256)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

freq_input = torch.randn(1, 3, 224, 224).to(device)
freq_cnn = FrequencyCNN().to(device)

with torch.no_grad():
    freq_output = freq_cnn(freq_input)
    print(f"Frequency CNN output shape: {freq_output.shape}")

# Test fusion
class FusionModule(nn.Module):
    def __init__(self, spatial_dim=768, freq_dim=256):
        super().__init__()
        self.spatial_proj = nn.Linear(spatial_dim, 512)
        self.freq_proj = nn.Linear(freq_dim, 512)
        self.fusion = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 2)
        )
    
    def forward(self, spatial_feat, freq_feat):
        print(f"Fusion input - spatial: {spatial_feat.shape}, freq: {freq_feat.shape}")
        spatial = self.spatial_proj(spatial_feat)
        freq = self.freq_proj(freq_feat)
        combined = torch.cat([spatial, freq], dim=1)
        return self.fusion(combined)

# Test fusion
fusion = FusionModule().to(device)
with torch.no_grad():
    fusion_output = fusion(flattened, freq_output)
    print(f"Fusion output shape: {fusion_output.shape}")

print("\n" + "="*50)
print("DIMENSION TEST RESULTS:")
print(f"Spatial features: {flattened.shape}")
print(f"Frequency features: {freq_output.shape}")
print(f"Fusion output: {fusion_output.shape}")
print("All shapes compatible!" if flattened.dim() == 2 and freq_output.dim() == 2 else "Shape mismatch!")
