import os
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class DocumentDenoisingData(Dataset):
    
    def __init__(self, root_dir:str, transform:transforms.Compose=None, target_transform:transforms.Compose=None):
        
        self.input_transform = transform
        self.target_transform = target_transform

        shabby_path = root_dir + '/shabby'
        clean_path = root_dir + '/cleaned'
        
        self.images = []
        self.targets = []

        for file in os.listdir(shabby_path):
            self.images.append(os.path.join(shabby_path, file))
            self.targets.append(os.path.join(clean_path, file))

        self.length = len(self.images)
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.item()

        img_path = self.images[idx]
        target_path = self.targets[idx]
        
        input_img = Image.open(img_path)
        target_img = Image.open(target_path)
        
        if self.input_transform:
            input_img = self.input_transform(input_img)
        
        if self.target_transform:
            target_img = self.target_transform(target_img)
        
        return {
            'Input' : input_img,
            'Target' : target_img
        }