
from torchvision import transforms
from torchvision.transforms import v2
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
import os
import pickle as pkl
import random
# Dataset Class
class RealFakeTripletDataset(Dataset):
    def __init__(self, root_dir, config,split="train"):
        self.real_path = os.path.join(root_dir, split, "real")
        self.fake_path = os.path.join(root_dir, split, "fake")
        self.real_files = [f for f in os.listdir(self.real_path) if f.endswith((".jpg", ".png"))]
        
        with open(f'{root_dir}/real2fake.pkl', 'rb') as f:
            self.real2fake = pkl.load(f)

        self.fake_files = [self.real2fake[i] for i in self.real_files]

        
        if config.debug:
            num_samples = config.get('batch_size') * 4
            self.real_files = self.real_files[:num_samples]
            self.fake_files = self.fake_files[:num_samples]
            
        custom_transforms = v2.Compose([
        v2.RandomResizedCrop((config.img_size,config.img_size), antialias=True),  # Randomly crop and resize the image to the desired size

        v2.RandomOrder([
            v2.RandomApply([v2.RandomChannelPermutation()], p=0.3),  # 30% chance
            v2.RandomApply([v2.RandomHorizontalFlip()], p=0.4),       # 50% chance
            v2.RandomApply([v2.RandomVerticalFlip()], p=0.4),         # 50% chance
            v2.RandomApply([v2.ColorJitter(brightness=0.2, contrast=0.2,
                                     saturation=0.2, hue=0.1)], p=0.5),  # 50% chance
            v2.RandomApply([v2.JPEG(quality=(75, 85))], p=0.4),
            # v2.RandomApply([v2.GaussianNoise()], p=0.3),
            v2.RandomApply([v2.GaussianBlur(5, sigma=0.5)], p=0.2)# 30% chance
        ]) ])  # Ensure that RandomOrder is always applied

        
 
        self.real_transform = v2.Compose([
                                    v2.RandomChoice([
                                            v2.RandomChoice([v2.AutoAugment(), #imagenet augment
                                               v2.RandAugment(), 
                                               v2.TrivialAugmentWide(),
                                               v2.AugMix()]),
                                            custom_transforms]),
                                    v2.Resize((config.img_size, config.img_size)),
                                    v2.ToTensor(),
                                    v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])
            
            
        
        self.fake_transform = v2.Compose([
            v2.RandomOrder([
                # v2.RandomApply([v2.RandomChannelPermutation()], p=0.1),  # 30% chance
                v2.RandomApply([v2.RandomHorizontalFlip()], p=0.1),       # 50% chance
                v2.RandomApply([v2.RandomVerticalFlip()], p=0.1),         # 50% chance
                v2.RandomApply([v2.ColorJitter(brightness=0.1, contrast=0.1,
                                         saturation=0.2, hue=0.1)], p=0.2),
                v2.RandomApply([v2.JPEG(quality=(75, 95))], p=0.4),
                v2.RandomApply([v2.JPEG(quality=(75, 85))], p=0.2)


            ]),
            v2.Resize((config.img_size, config.img_size)),
            v2.ToTensor(),
            v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
    def __len__(self):
        return len(self.real_files)
    def load_image(self, path):
        """Custom image loader with error handling"""
        
        return Image.open(path).convert('RGB')
        
   

    def __getitem__(self, idx):
            anchor_real_path = os.path.join(self.real_path, self.real_files[idx])
            anchor = self.load_image(anchor_real_path)

            # Select a positive: another real image, ensuring it's not the anchor
            positive_idx = idx
            while positive_idx == idx:
                positive_idx = random.randint(0, len(self.real_files) - 1)
            positive_real_path = os.path.join(self.real_path, self.real_files[positive_idx])
            positive = self.load_image(positive_real_path)

            # Negative: fake counterpart of the anchor
            fake_filename = self.fake_files[idx]
            negative_fake_path = os.path.join(self.fake_path, fake_filename)
            negative = self.load_image(negative_fake_path)

            return {
                "anchor": self.real_transform(anchor),
                "positive": self.real_transform(positive),
                "negative": self.fake_transform(negative)
            }



class RealFakeShuffledDataset(RealFakeTripletDataset):
    """
    Dataset class for classification tasks that inherits from RealFakeTripletDataset.
    It provides individual images along with their corresponding labels.
    
    Labels:
        - 0: Real Image
        - 1: Fake Image
    """
    def __init__(self, root_dir, config, split="train"):
        """
        Initializes the RealFakeShuffledDataset.
        
        Args:
            root_dir (str): Root directory of the dataset.
            config (obj): Configuration object containing various settings.
            split (str): Indicates the dataset split ('train', 'val', etc.).
        """
        super().__init__(root_dir, config, split)
        
        # Combine real and fake images into a single list
        self.images = self.real_files + self.fake_files
        # Create corresponding labels: 0 for real, 1 for fake
        self.labels = [0] * len(self.real_files) + [1] * len(self.fake_files)
    
    def __len__(self):
        """
        Returns the total number of samples in the dataset.
        """
        return len(self.images)
    
    def __getitem__(self, idx):
        """
        Retrieves the image and its label at the specified index.
        
        Args:
            idx (int): Index of the sample to retrieve.
        
        Returns:
            tuple: (image_tensor, label)
        """
        img_filename = self.images[idx]
        label = self.labels[idx]
        
        if label == 0:
            img_path = os.path.join(self.real_path, img_filename)
            transform = self.real_transform
        else:
            img_path = os.path.join(self.fake_path, img_filename)
            transform = self.fake_transform
        
        img = transform(self.load_image(img_path))

        
        return img, label




    
    
