import torch
import timm
from torchvision import transforms
from PIL import Image
import os

# Load Swin-T model
model = timm.create_model('swin_tiny_patch4_window7_224', pretrained=True)
model.eval()

# Load a random image from dataset
dataset_path = "datasets/train/real"
image_files = [f for f in os.listdir(dataset_path) if f.endswith(('.jpg', '.jpeg'))]
image_path = os.path.join(dataset_path, image_files[0])

# Preprocess image
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

image = Image.open(image_path).convert('RGB')
image_tensor = transform(image).unsqueeze(0)

# Test forward pass
with torch.no_grad():
    output = model(image_tensor)
    print(f"Model loaded successfully!")
    print(f"Input shape: {image_tensor.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Test passed - Swin-T works!")
