# Deepfake Detection Training - Colab Notebook
# Copy and paste each cell into separate Colab notebook cells

# ===== CELL 1: INSTALLATION & SETUP =====
!pip install torch torchvision timm opencv-python scikit-image matplotlib seaborn tqdm tensorboard

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from torch.utils.data import DataLoader
from torchvision import transforms
import cv2
import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import time

# Check GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# ===== CELL 2: MOUNT GOOGLE DRIVE =====
from google.colab import drive
drive.mount('/content/drive')

# Update this path to where your dataset is in Drive
DATASET_PATH = "/content/drive/MyDrive/deepfake_datasets"
SAVE_PATH = "/content/drive/MyDrive/deepfake_models"

os.makedirs(SAVE_PATH, exist_ok=True)

# ===== CELL 3: MODEL ARCHITECTURE =====
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
        # Load Swin-T backbone
        self.swin_backbone = timm.create_model('swin_tiny_patch4_window7_224', pretrained=True)
        
        # Get the feature dimension from the original head
        self.feature_dim = self.swin_backbone.head.fc.in_features
        
        # Remove classification head
        self.swin_backbone.head = nn.Identity()
        
        self.freq_cnn = FrequencyCNN()
        self.fusion = FusionModule(spatial_dim=self.feature_dim)
    
    def forward(self, spatial_input, freq_input):
        spatial_feat = self.swin_backbone(spatial_input)  # [batch, 7, 7, feature_dim]
        
        # Global average pool over spatial dimensions (H, W) to get [batch, feature_dim]
        if spatial_feat.dim() == 4:
            spatial_feat = torch.mean(spatial_feat, dim=[1, 2])  # [batch, feature_dim]
        
        freq_feat = self.freq_cnn(freq_input)        # [batch, 256]
        
        return self.fusion(spatial_feat, freq_feat)

# ===== CELL 4: DATA PIPELINE =====
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from enum import Enum

class FrequencyType(Enum):
    DCT = "dct"
    FFT = "fft"

class FrequencyExtractor:
    def __init__(self, frequency_type: FrequencyType):
        self.frequency_type = frequency_type
    
    def extract(self, image: np.ndarray) -> np.ndarray:
        if self.frequency_type == FrequencyType.DCT:
            return self._extract_dct(image)
        elif self.frequency_type == FrequencyType.FFT:
            return self._extract_fft(image)
        else:
            raise ValueError(f"Unsupported frequency type: {self.frequency_type}")
    
    def _extract_dct(self, image: np.ndarray) -> np.ndarray:
        ycbcr = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
        dct = cv2.dct(ycbcr[:, :, 0].astype(np.float32))
        return dct
    
    def _extract_fft(self, image: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        fft = np.fft.fft2(gray)
        magnitude = np.abs(fft)
        magnitude = np.fft.fftshift(magnitude)
        return magnitude

class DeepfakeDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir: str, transform=None, frequency_type=FrequencyType.DCT, 
                 image_size=(224, 224), cache_frequency=False):  # Default to NO caching
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.frequency_extractor = FrequencyExtractor(frequency_type)
        self.image_size = image_size
        self.cache_frequency = cache_frequency  # Disabled by default to save RAM
        self.frequency_cache = {} if cache_frequency else None
        
        self.samples = []
        self._load_samples()
    
    def _load_samples(self):
        for label, class_name in enumerate(['real', 'fake']):
            class_dir = self.root_dir / class_name
            if not class_dir.exists():
                continue
            
            for img_path in class_dir.glob('*'):
                if img_path.suffix.lower() in {'.jpg', '.jpeg', '.png'}:
                    self.samples.append((img_path, label))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        # Load spatial image
        image = Image.open(img_path).convert('RGB')
        spatial_tensor = self._process_spatial(image)
        
        # Extract frequency features
        freq_tensor = self._process_frequency(img_path, image)
        
        return spatial_tensor, freq_tensor, label
    
    def _process_spatial(self, image):
        if self.transform:
            image = self.transform(image)
        else:
            image = transforms.ToTensor()(image)
        return image
    
    def _process_frequency(self, img_path, image):
        # Always compute on-the-fly to save RAM (no caching)
        image_np = np.array(image)
        freq_features = self.frequency_extractor.extract(image_np)
        
        # Normalize and convert to tensor
        freq_features = np.log1p(np.abs(freq_features))
        freq_min, freq_max = freq_features.min(), freq_features.max()
        if freq_max > freq_min:
            freq_features = (freq_features - freq_min) / (freq_max - freq_min)
        
        freq_tensor = torch.from_numpy(freq_features).float()
        
        # Ensure 3 channels
        if freq_tensor.dim() == 2:
            freq_tensor = freq_tensor.unsqueeze(0).repeat(3, 1, 1)
        
        # Resize to match spatial dimensions
        freq_tensor = F.interpolate(
            freq_tensor.unsqueeze(0), 
            size=self.image_size, 
            mode='bilinear', 
            align_corners=False
        ).squeeze(0)
        
        return freq_tensor

# ===== CELL 5: DATA TRANSFORMS AND LOADERS =====
# Transforms
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.RandomRotation(degrees=10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Create datasets
train_dataset = DeepfakeDataset(
    root_dir=f"{DATASET_PATH}/train",
    transform=train_transform,
    frequency_type=FrequencyType.DCT
)

val_dataset = DeepfakeDataset(
    root_dir=f"{DATASET_PATH}/val",
    transform=val_transform,
    frequency_type=FrequencyType.DCT
)

print(f"Training samples: {len(train_dataset)}")
print(f"Validation samples: {len(val_dataset)}")

# Data loaders - ULTRA STABLE for Colab (minimal RAM usage)
batch_size = 8  # Reduced from 32 to save RAM
accumulation_steps = 4  # Keep effective batch size at 32

# num_workers=0, no pin_memory, no caching to minimize RAM usage
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                         num_workers=0, pin_memory=False, drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, 
                       num_workers=0, pin_memory=False, drop_last=False)

print(f"Batch size: {batch_size}")
print(f"Accumulation steps: {accumulation_steps}")
print(f"Effective batch size: {batch_size * accumulation_steps}")
print(f"RAM optimization: Frequency caching DISABLED")

# ===== CELL 6: TRAINING SETUP =====
from torch.amp import autocast  # Updated import for newer PyTorch versions
from torch.cuda.amp import GradScaler

# Initialize model
model = DeepfakeDetector().to(device)

# Enable DataParallel if multiple GPUs available
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs!")
    model = nn.DataParallel(model)

# Mixed precision training for memory efficiency and speed
scaler = GradScaler()

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)  # Increased LR for larger batch
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

# Training parameters
num_epochs = 100  # Increased epochs with better scheduler
best_val_acc = 0.0
patience = 15  # Increased patience
patience_counter = 0

# Memory optimization
print(f"GPU Memory allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB")
print(f"GPU Memory reserved: {torch.cuda.memory_reserved()/1e9:.2f} GB")
print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

# Set device for autocast
AMP_DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# ===== CELL 7: TRAINING LOOP =====
train_losses = []
val_losses = []
train_accs = []
val_accs = []

for epoch in range(num_epochs):
    # Aggressive memory cleanup at start of epoch
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    
    # Training
    model.train()
    train_loss = 0.0
    correct = 0
    total = 0
    
    train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]')
    optimizer.zero_grad()  # Zero gradients at start of epoch
    
    for batch_idx, (spatial_imgs, freq_imgs, labels) in enumerate(train_pbar):
        spatial_imgs = spatial_imgs.to(device)
        freq_imgs = freq_imgs.to(device)
        labels = labels.to(device)
        
        # Mixed precision forward pass
        with autocast(device_type=AMP_DEVICE):
            outputs = model(spatial_imgs, freq_imgs)
            loss = criterion(outputs, labels) / accumulation_steps  # Normalize loss
        
        # Scale loss and backprop
        scaler.scale(loss).backward()
        
        # Update weights only after accumulation_steps
        if (batch_idx + 1) % accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        
        train_loss += loss.item() * accumulation_steps  # Unnormalize for logging
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        train_pbar.set_postfix({
            'loss': loss.item() * accumulation_steps, 
            'acc': 100.*correct/total,
            'gpu_mem': f"{torch.cuda.memory_allocated()/1e9:.1f}GB"
        })
        
        # Periodic memory cleanup every 100 batches to prevent RAM buildup
        if (batch_idx + 1) % 100 == 0:
            import gc
            gc.collect()
            torch.cuda.empty_cache()
    
    train_loss = train_loss / len(train_loader)
    train_acc = 100 * correct / total
    
    # Handle any remaining gradients at end of epoch
    if (batch_idx + 1) % accumulation_steps != 0:
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
    
    # Validation
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for spatial_imgs, freq_imgs, labels in tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Val]'):
            spatial_imgs = spatial_imgs.to(device)
            freq_imgs = freq_imgs.to(device)
            labels = labels.to(device)
            
            # Mixed precision for validation too
            with autocast(device_type=AMP_DEVICE):
                outputs = model(spatial_imgs, freq_imgs)
                loss = criterion(outputs, labels)
            
            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    val_loss = val_loss / len(val_loader)
    val_acc = 100 * correct / total
    
    # Update scheduler
    scheduler.step()
    
    # Clear cache to prevent memory buildup
    torch.cuda.empty_cache()
    
    # Additional RAM cleanup
    import gc
    gc.collect()
    
    # Save metrics
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    train_accs.append(train_acc)
    val_accs.append(val_acc)
    
    # Print epoch results
    print(f'Epoch {epoch+1}/{num_epochs}:')
    print(f'  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
    print(f'  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
    print(f'  LR: {scheduler.get_last_lr()[0]:.6f}')
    
    # Save best model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        patience_counter = 0
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'val_acc': val_acc,
            'val_loss': val_loss,
        }, f'{SAVE_PATH}/best_model.pth')
        print(f'  New best model saved! Val Acc: {val_acc:.2f}%')
    else:
        patience_counter += 1
    
    # Early stopping
    if patience_counter >= patience:
        print(f'Early stopping triggered after {epoch+1} epochs')
        break
    
    print('-' * 50)

print(f'Training completed! Best Val Acc: {best_val_acc:.2f}%')

# Clear all memory after training
import gc
gc.collect()
torch.cuda.empty_cache()

# ===== CELL 8: VISUALIZE RESULTS =====
plt.figure(figsize=(15, 5))

# Loss plot
plt.subplot(1, 3, 1)
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Val Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# Accuracy plot
plt.subplot(1, 3, 2)
plt.plot(train_accs, label='Train Acc')
plt.plot(val_accs, label='Val Acc')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.legend()

# Learning rate plot
plt.subplot(1, 3, 3)
lrs = [scheduler.get_last_lr()[0] for _ in range(len(train_losses))]
plt.plot(lrs)
plt.title('Learning Rate Schedule')
plt.xlabel('Epoch')
plt.ylabel('Learning Rate')

plt.tight_layout()
plt.show()

# ===== CELL 9: TEST BEST MODEL =====
# Load best model
checkpoint = torch.load(f'{SAVE_PATH}/best_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])

# Evaluate on validation set
model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for spatial_imgs, freq_imgs, labels in tqdm(val_loader, desc='Final Evaluation'):
        spatial_imgs = spatial_imgs.to(device)
        freq_imgs = freq_imgs.to(device)
        
        outputs = model(spatial_imgs, freq_imgs)
        _, preds = torch.max(outputs, 1)
        
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())

from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import seaborn as sns

# Print classification report
print("Classification Report:")
print(classification_report(all_labels, all_preds, target_names=['Real', 'Fake']))

# Calculate AUC-ROC
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc

# Get probabilities for AUC
model.eval()
all_probs = []
with torch.no_grad():
    for spatial_imgs, freq_imgs, labels in tqdm(val_loader, desc='Getting Probabilities'):
        spatial_imgs = spatial_imgs.to(device)
        freq_imgs = freq_imgs.to(device)
        
        outputs = model(spatial_imgs, freq_imgs)
        probs = torch.softmax(outputs, dim=1)[:, 1]  # Probability of fake
        all_probs.extend(probs.cpu().numpy())

# Binarize labels for ROC
y_test_bin = label_binarize(all_labels, classes=[0, 1])
fpr, tpr, _ = roc_curve(y_test_bin, all_probs)
roc_auc = auc(fpr, tpr)

print(f"AUC-ROC: {roc_auc:.4f}")

# Plot confusion matrix
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Real', 'Fake'], yticklabels=['Real', 'Fake'])
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()

# Plot ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()

print(f"Final Results:")
print(f"Best Validation Accuracy: {best_val_acc:.2f}%")
print(f"AUC-ROC: {roc_auc:.4f}")
print(f"Model saved at: {SAVE_PATH}/best_model.pth")

# ===== CELL 10: SAMPLE PREDICTIONS =====
def predict_sample(model, spatial_img, freq_img, true_label):
    model.eval()
    with torch.no_grad():
        spatial_img = spatial_img.unsqueeze(0).to(device)
        freq_img = freq_img.unsqueeze(0).to(device)
        
        output = model(spatial_img, freq_img)
        prob = torch.softmax(output, dim=1)
        pred_label = torch.argmax(prob).item()
        confidence = prob.max().item()
        
        return pred_label, confidence

# Get some sample predictions
samples = iter(val_loader)
spatial_batch, freq_batch, labels_batch = next(samples)

plt.figure(figsize=(15, 10))
for i in range(min(8, len(spatial_batch))):
    plt.subplot(2, 4, i+1)
    
    # Denormalize image for display
    img = spatial_batch[i].cpu()
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img = img * std + mean
    img = torch.clamp(img, 0, 1)
    
    plt.imshow(img.permute(1, 2, 0))
    
    pred_label, confidence = predict_sample(model, spatial_batch[i], freq_batch[i], labels_batch[i])
    true_label = labels_batch[i].item()
    
    color = 'green' if pred_label == true_label else 'red'
    title = f'True: {"Real" if true_label == 0 else "Fake"}\n'
    title += f'Pred: {"Real" if pred_label == 0 else "Fake"} ({confidence:.2f})'
    plt.title(title, color=color)
    plt.axis('off')

plt.tight_layout()
plt.show()

print("Training completed! Check your Google Drive for the saved model and results.")
