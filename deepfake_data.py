from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import cv2
import numpy as np
from enum import Enum


class FrequencyType(Enum):
    DCT = "dct"
    FFT = "fft"


@dataclass
class DatasetConfig:
    root_dir: Union[str, Path]
    batch_size: int = 32
    num_workers: int = 4
    frequency_type: FrequencyType = FrequencyType.DCT
    image_size: Tuple[int, int] = (224, 224)
    pin_memory: bool = True
    shuffle_train: bool = True
    drop_last: bool = True


@dataclass
class FrequencyFeatures:
    dct_coefficients: Optional[np.ndarray] = None
    fft_magnitude: Optional[np.ndarray] = None
    spatial_features: Optional[np.ndarray] = None


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


class DeepfakeDataset(Dataset):
    def __init__(
        self,
        root_dir: Union[str, Path],
        transform: Optional[transforms.Compose] = None,
        frequency_type: FrequencyType = FrequencyType.DCT,
        cache_frequency: bool = True,
        image_size: Tuple[int, int] = (224, 224)
    ):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.frequency_extractor = FrequencyExtractor(frequency_type)
        self.cache_frequency = cache_frequency
        self.image_size = image_size
        self.frequency_cache: Dict[str, np.ndarray] = {}
        
        self.samples: List[Tuple[Path, int]] = []
        self._load_samples()
    
    def _load_samples(self) -> None:
        for label, class_name in enumerate(['real', 'fake']):
            class_dir = self.root_dir / class_name
            if not class_dir.exists():
                continue
            
            for img_path in class_dir.glob('*'):
                if img_path.suffix.lower() in {'.jpg', '.jpeg', '.png'}:
                    self.samples.append((img_path, label))
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        img_path, label = self.samples[idx]
        
        # Load spatial image
        image = Image.open(img_path).convert('RGB')
        spatial_tensor = self._process_spatial(image)
        
        # Extract frequency features
        freq_tensor = self._process_frequency(img_path, image)
        
        return spatial_tensor, freq_tensor, label
    
    def _process_spatial(self, image: Image.Image) -> torch.Tensor:
        if self.transform:
            image = self.transform(image)
        else:
            image = transforms.ToTensor()(image)
        return image
    
    def _process_frequency(self, img_path: Path, image: Image.Image) -> torch.Tensor:
        cache_key = str(img_path)
        
        if self.cache_frequency and cache_key in self.frequency_cache:
            freq_features = self.frequency_cache[cache_key]
        else:
            image_np = np.array(image)
            freq_features = self.frequency_extractor.extract(image_np)
            
            if self.cache_frequency:
                self.frequency_cache[cache_key] = freq_features
        
        # Normalize and convert to tensor
        freq_features = self._normalize_frequency(freq_features)
        freq_tensor = torch.from_numpy(freq_features).float()
        
        # Ensure 3 channels for CNN compatibility
        if freq_tensor.dim() == 2:
            freq_tensor = freq_tensor.unsqueeze(0).repeat(3, 1, 1)
        
        # Resize to match spatial dimensions
        freq_tensor = torch.nn.functional.interpolate(
            freq_tensor.unsqueeze(0), 
            size=self.image_size, 
            mode='bilinear', 
            align_corners=False
        ).squeeze(0)
        
        return freq_tensor
    
    def _normalize_frequency(self, freq_features: np.ndarray) -> np.ndarray:
        # Log transform for better dynamic range
        freq_features = np.log1p(np.abs(freq_features))
        
        # Normalize to [0, 1]
        freq_min, freq_max = freq_features.min(), freq_features.max()
        if freq_max > freq_min:
            freq_features = (freq_features - freq_min) / (freq_max - freq_min)
        
        return freq_features


class DataModule:
    def __init__(self, config: DatasetConfig):
        self.config = config
        self.train_transform = transforms.Compose([
            transforms.Resize(config.image_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.RandomRotation(degrees=10),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        self.val_transform = transforms.Compose([
            transforms.Resize(config.image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    def setup(self, train_dir: Union[str, Path], val_dir: Union[str, Path]) -> None:
        self.train_dataset = DeepfakeDataset(
            root_dir=train_dir,
            transform=self.train_transform,
            frequency_type=self.config.frequency_type,
            cache_frequency=True,
            image_size=self.config.image_size
        )
        
        self.val_dataset = DeepfakeDataset(
            root_dir=val_dir,
            transform=self.val_transform,
            frequency_type=self.config.frequency_type,
            cache_frequency=True,
            image_size=self.config.image_size
        )
    
    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=self.config.shuffle_train,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            drop_last=self.config.drop_last
        )
    
    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            drop_last=False
        )
    
    def get_sample_batch(self, num_samples: int = 4) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        train_loader = self.train_dataloader()
        batch = next(iter(train_loader))
        return batch[0][:num_samples], batch[1][:num_samples], batch[2][:num_samples]


