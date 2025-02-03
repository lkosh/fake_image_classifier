from torchvision import transforms
from torchvision.transforms import v2
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
import os
import pickle as pkl
# Dataset Class
class RealFakeDataset(Dataset):
    def __init__(self, root_dir, config,split="train"):
        self.real_path = os.path.join(root_dir, split, "real")
        self.fake_path = os.path.join(root_dir, split, "fake")
        self.real_files = [f for f in os.listdir(self.real_path) if f.endswith((".jpg", ".png"))]
        
        with open(f'{root_dir}/real2fake.pkl', 'rb') as f:
            self.real2fake = pkl.load(f)

        self.fake_files = [self.real2fake[i] for i in self.real_files]

        if config.debug:
            self.real_files = self.real_files[:5]
            self.fake_files = self.fake_files[:5]
            
        custom_transforms = v2.Compose([
        v2.RandomResizedCrop((config.img_size,config.img_size), antialias=True),  # Randomly crop and resize the image to the desired size

        v2.RandomOrder([
            v2.RandomApply([v2.RandomChannelPermutation()], p=0.1),  # 30% chance
            v2.RandomApply([v2.RandomHorizontalFlip()], p=0.2),       # 50% chance
            v2.RandomApply([v2.RandomVerticalFlip()], p=0.2),         # 50% chance
            v2.RandomApply([v2.ColorJitter(brightness=0.2, contrast=0.2,
                                     saturation=0.2, hue=0.1)], p=0.3),  # 50% chance
            v2.RandomApply([v2.JPEG(quality=(75, 85))], p=0.2),
            # v2.RandomApply([v2.GaussianNoise()], p=0.3),
            v2.RandomApply([v2.GaussianBlur(5, sigma=0.5)], p=0.05)# 30% chance
        ]) ])  # Ensure that RandomOrder is always applied

        # self.real_transform = T.Compose([
        #     T.Resize((config.img_size, config.img_size)),
        #     T.RandomHorizontalFlip(),
        #     T.ColorJitter(0.1, 0.1, 0.1),
        #     T.ToTensor(),
        #     T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) #vit expects normalized inputs

        # ])
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
                v2.RandomApply([v2.RandomHorizontalFlip()], p=0.05),       # 50% chance
                v2.RandomApply([v2.RandomVerticalFlip()], p=0.05),         # 50% chance
                v2.RandomApply([v2.ColorJitter(brightness=0.1, contrast=0.1,
                                         saturation=0.2, hue=0.1)], p=0.2),
                v2.RandomApply([v2.JPEG(quality=(75, 95))], p=0.1),

            ]),
            v2.Resize((config.img_size, config.img_size)),
            v2.ToTensor(),
            v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
    def __len__(self):
        return len(self.real_files)
    def load_image(self, path, transform):
        """Custom image loader with error handling"""
        
        img = Image.open(path).convert('RGB')
        return transform(img)
   
    def __getitem__(self, idx):
        real_img = self.load_image(os.path.join(self.real_path, self.real_files[idx]), self.real_transform)
        fake_img = self.load_image(os.path.join(self.fake_path, self.fake_files[idx]), self.fake_transform)
        
        return {
            "real": real_img,
            "fake": fake_img
        }




    