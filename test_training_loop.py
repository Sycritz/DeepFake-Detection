import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm
import os

# Simple test dataset with just a few images
class TestDataset(torch.utils.data.Dataset):
    def __init__(self, num_samples=8):
        self.num_samples = num_samples
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # Create dummy images as PIL Images
        spatial_array = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
        freq_array = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
        label = np.random.randint(0, 2)
        
        spatial_img = Image.fromarray(spatial_array)
        spatial_img = self.transform(spatial_img)
        
        # Convert frequency to tensor directly
        freq_img = torch.from_numpy(freq_array).float() / 255.0
        freq_img = freq_img.permute(2, 0, 1)  # HWC -> CHW
        
        return spatial_img, freq_img, label

# Model components
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
        print(f"Fusion input shapes - spatial: {spatial_feat.shape}, freq: {freq_feat.shape}")
        spatial = self.spatial_proj(spatial_feat)
        freq = self.freq_proj(freq_feat)
        combined = torch.cat([spatial, freq], dim=1)
        print(f"Combined shape: {combined.shape}")
        return self.fusion(combined)

class DeepfakeDetector(nn.Module):
    def __init__(self):
        super().__init__()
        # Load Swin-T backbone
        self.swin_backbone = timm.create_model('swin_tiny_patch4_window7_224', pretrained=True)
        
        # Get the feature dimension before the head
        self.feature_dim = self.swin_backbone.head.fc.in_features
        print(f"Expected feature dimension: {self.feature_dim}")
        
        # Remove the head completely
        self.swin_backbone.head = nn.Identity()
        
        self.freq_cnn = FrequencyCNN()
        self.fusion = FusionModule(spatial_dim=self.feature_dim)
    
    def forward(self, spatial_input, freq_input):
        spatial_feat = self.swin_backbone(spatial_input)  # [batch, H, W, feature_dim]
        print(f"Raw spatial feat shape: {spatial_feat.shape}")
        
        # Global average pool over H and W dimensions to get [batch, feature_dim]
        if spatial_feat.dim() == 4:
            # Pool over H and W (last two dimensions)
            spatial_feat = torch.mean(spatial_feat, dim=[1, 2])  # [batch, feature_dim]
        
        print(f"Processed spatial feat shape: {spatial_feat.shape}")
        
        freq_feat = self.freq_cnn(freq_input)        # [batch, 256]
        print(f"Frequency feat: {freq_feat.shape}")
        
        return self.fusion(spatial_feat, freq_feat)

# Test training loop
def test_training():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create model
    model = DeepfakeDetector().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    # Create test dataset
    train_dataset = TestDataset(num_samples=8)
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Training samples: {len(train_dataset)}")
    
    # Test one forward pass
    model.train()
    for batch_idx, (spatial_imgs, freq_imgs, labels) in enumerate(train_loader):
        print(f"\nBatch {batch_idx}:")
        print(f"Spatial batch shape: {spatial_imgs.shape}")
        print(f"Frequency batch shape: {freq_imgs.shape}")
        print(f"Labels shape: {labels.shape}")
        
        spatial_imgs = spatial_imgs.to(device)
        freq_imgs = freq_imgs.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        try:
            outputs = model(spatial_imgs, freq_imgs)
            print(f"Model output shape: {outputs.shape}")
            
            loss = criterion(outputs, labels)
            print(f"Loss: {loss.item():.4f}")
            
            loss.backward()
            optimizer.step()
            
            # Calculate accuracy
            _, predicted = torch.max(outputs.data, 1)
            total = labels.size(0)
            correct = (predicted == labels).sum().item()
            accuracy = 100 * correct / total
            print(f"Batch accuracy: {accuracy:.2f}%")
            
            print("✅ Training step successful!")
            break  # Just test one batch
            
        except Exception as e:
            print(f"❌ Error in training step: {e}")
            import traceback
            traceback.print_exc()
            break

if __name__ == "__main__":
    test_training()
