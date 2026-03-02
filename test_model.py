import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from pathlib import Path
import cv2

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
        spatial = self.spatial_proj(spatial_feat)
        freq = self.freq_proj(freq_feat)
        combined = torch.cat([spatial, freq], dim=1)
        return self.fusion(combined)

class DeepfakeDetector(nn.Module):
    def __init__(self):
        super().__init__()
        self.swin_backbone = timm.create_model('swin_tiny_patch4_window7_224', pretrained=True)
        self.feature_dim = self.swin_backbone.head.fc.in_features
        self.swin_backbone.head = nn.Identity()
        self.freq_cnn = FrequencyCNN()
        self.fusion = FusionModule(spatial_dim=self.feature_dim)
    
    def forward(self, spatial_input, freq_input):
        spatial_feat = self.swin_backbone(spatial_input)
        if spatial_feat.dim() == 4:
            spatial_feat = torch.mean(spatial_feat, dim=[1, 2])
        freq_feat = self.freq_cnn(freq_input)
        return self.fusion(spatial_feat, freq_feat)

def extract_dct(image_np):
    ycbcr = cv2.cvtColor(image_np, cv2.COLOR_RGB2YCrCb)
    dct = cv2.dct(ycbcr[:, :, 0].astype(np.float32))
    dct = np.log1p(np.abs(dct))
    dct_min, dct_max = dct.min(), dct.max()
    if dct_max > dct_min:
        dct = (dct - dct_min) / (dct_max - dct_min)
    return dct

def extract_fft(image_np):
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    fft = np.fft.fft2(gray)
    magnitude = np.abs(fft)
    magnitude = np.fft.fftshift(magnitude)
    magnitude = np.log1p(magnitude)
    mag_min, mag_max = magnitude.min(), magnitude.max()
    if mag_max > mag_min:
        magnitude = (magnitude - mag_min) / (mag_max - mag_min)
    return magnitude

def test_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = DeepfakeDetector().to(device)
    checkpoint = torch.load('best_model.pth', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    test_dir = Path('test-images')
    if not test_dir.exists():
        print(f"Directory {test_dir} not found")
        return
    
    image_files = list(test_dir.glob('*.jpg')) + list(test_dir.glob('*.png')) + list(test_dir.glob('*.jpeg'))
    
    if not image_files:
        print(f"No images found in {test_dir}")
        return
    
    print(f"Testing {len(image_files)} images...")
    print("-" * 40)
    
    with torch.no_grad():
        for img_path in image_files:
            image = Image.open(img_path).convert('RGB')
            spatial_tensor = transform(image).unsqueeze(0).to(device)
            
            # Extract DCT features (can switch to extract_fft if needed)
            image_np = np.array(image.resize((224, 224)))
            freq_features = extract_dct(image_np)
            
            # Convert to 3-channel tensor
            freq_tensor = torch.from_numpy(freq_features).float()
            freq_tensor = freq_tensor.unsqueeze(0).unsqueeze(0).repeat(1, 3, 1, 1).to(device)
            
            output = model(spatial_tensor, freq_tensor)
            probs = torch.softmax(output, dim=1)
            pred = torch.argmax(probs).item()
            confidence = probs.max().item()
            
            label = "Real" if pred == 0 else "Fake"
            print(f"{img_path.name}: {label} ({confidence:.2%})")

if __name__ == "__main__":
    test_model()
