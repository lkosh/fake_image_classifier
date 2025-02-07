import os
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import ViTModel, ViTConfig
import numpy as np
from facenet_pytorch import MTCNN
from PIL import Image
import evaluate
import pickle as pkl
import json
# from torch.cuda.amp import GradScaler, autocast

# Configuration


import torch
from freqnet import freqnet
from torch.nn import TransformerEncoder, TransformerEncoderLayer

from dataset import RealFakeTripletDataset, RealFakeShuffledDataset



class SimpleFusion(nn.Module):
    def __init__(self, feature_dim=256, num_heads=8, num_layers=2, dropout=0.1):
        super(SimpleFusion, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_features=3 * feature_dim, out_features=feature_dim),
            nn.ReLU()
            
        )

    def forward(self, vit, facenet, freqnet):
        # Combine features into a sequence
        combined = torch.cat((vit, facenet, freqnet), dim=1)
        out = self.fc(combined)
        return out
        
class FeatureProjection(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(FeatureProjection, self).__init__()
        self.proj = nn.Linear(input_dim, output_dim)
        self.bn = nn.BatchNorm1d(output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.proj(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class PoolingFaceEncoder(nn.Module):
    def __init__(self, box_dim=4, keypoints_dim=10, hidden_dim=128):
        super(PoolingFaceEncoder, self).__init__()
        #add 1 dimension for probabilities
        input_dim = box_dim + keypoints_dim + 1  # 4 + 10 = 14
        self.hidden_dim = hidden_dim
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.pool = nn.AdaptiveAvgPool1d(1)  # Mean pooling

    def forward(self, boxes, probs, keypoints):
        """
        boxes: Tensor of shape (B, N, 4)
        keypoints: Tensor of shape (B, N, 5, 2)
        Returns:
            fixed_features: Tensor of shape (B, hidden_dim)
        """
        
        B, N, _ = boxes.shape
        # Flatten keypoints
        keypoints = keypoints.view(B, N, -1)  # (B, N, 10)
        probs = probs.view(B, N, -1)
        # Concatenate boxes and keypoints
        features = torch.cat([boxes, keypoints, probs], dim=-1)  # (B, N, 14)
        # Project each face feature
        features = self.mlp(features)  # (B, N, hidden_dim)
        # Transpose for pooling
        features = features.transpose(1, 2)  # (B, hidden_dim, N)
        # Apply mean pooling
        pooled = self.pool(features).squeeze(-1)  # (B, hidden_dim)
        # pooled = pooled.permute(0, 2, 1).view(B ,   self.hidden_dim)
        return pooled

# Hybrid Model
class ArtifactDetector(nn.Module):
    def __init__(self,
                 config,
                 gpu_id
                ):
        super().__init__()
        self.gpu_id = gpu_id
        
        # Vision Transformer branch
        self.vit = ViTModel.from_pretrained(config.vit_model)
        self.vit.config.use_memory_efficient_attention = True
        self.vit.config.gradient_checkpointing = True

        common_dim = config.common_dim
        facenet_dim = common_dim  #we got  2 embeddings
        freqnet_dim= config.freqnet_dim
        num_heads = config.num_heads
        dropout=config.dropout
        num_layers=config.num_layers
        vit_dim = self.vit.config.hidden_size
        self.config= config
        self.batch_size = config.batch_size
        #layers


        self.freqnet = freqnet()#has 512 channels
        #load pretrained weights
       
        #use this method to compute batch_points and batch_boxes
        self.mtcnn = MTCNN(image_size=config.img_size, keep_all=True, select_largest=False)
        self.face_net = PoolingFaceEncoder(hidden_dim=facenet_dim )
        # # Face branch

        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(common_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        # Auxiliary face detector
        self.face_aux = nn.Linear(facenet_dim, 1) #take only facenet embeddings

        self.vit_proj = FeatureProjection(vit_dim, common_dim)
        # self.facenet_proj = FeatureProjection(facenet_dim, common_dim) is already the correct size
        self.freqnet_proj = FeatureProjection(freqnet_dim, common_dim)

        #feature fusion
        self.feature_fusion = SimpleFusion(feature_dim=common_dim)
        self.init_weights()

    def init_weights(self):
        #initialize freqnet:
        state_dict = torch.load(self.config.freqnet_weights, map_location=f"cuda:{self.gpu_id}")
        self.freqnet.load_state_dict(state_dict, strict=False)

        #TODO initialize classification layer
        # for layer in self.classifier.modules():
        #     nn.init.xavier_uniform_(layer.weight)
        #     nn.init.zeros_(layer.bias)
        #pytorch initializes layers with uniform distr.
        #optional TODO: add more advanced weight init
      
        
    def forward(self, rgb):
        batch_dim = rgb.size(0)#
        # RGB branch
        vit_out = self.vit(rgb).last_hidden_state[:, 0]
        # vit_feat = self.vit_fc(vit_out)
        
        # Frequency branch
        fft_feat = self.freqnet(rgb)
        fft_feat = fft_feat.squeeze() #512 channels

        mtcnn_sus = False
        batch_boxes, batch_probs, batch_points = None, None, None
        # Face branch
        # face_feat = self.face_net(rgb).squeeze()
        #TODO mtcnn throws an error when input tensor is sus T_T why u do dis to me 
        #-1 embedding means mtcnn thinks the input image is sus, 0 just means "no faces found"
        try:
            batch_boxes, batch_probs, batch_points = self.mtcnn.detect(rgb)
        except RuntimeError as e:
            mtcnn_sus = True
        
        if batch_boxes == None:
            N = 2 #we got two images
            if not mtcnn_sus:
                #just a regular faceless image lol
                batch_boxes = torch.zeros((batch_dim, N, 4)).cuda(self.gpu_id) #N = 2
                batch_probs = torch.zeros((batch_dim, N)).cuda(self.gpu_id)
                batch_points = torch.zeros((batch_dim, N, 5, 2)).cuda(self.gpu_id)
            else:
                #mtcnn thinks this image is sus 
                batch_boxes = -1 * torch.ones((batch_dim, N, 4)).cuda(self.gpu_id) #N = 2
                batch_probs = -1 * torch.ones((batch_dim, N)).cuda(self.gpu_id)
                batch_points =-1 * torch.ones((batch_dim, N, 5, 2)).cuda(self.gpu_id)
                
        face_feat = self.face_net(batch_boxes, batch_probs, batch_points)

        vit_proj = self.vit_proj(vit_out)
        freqnet_feat = self.freqnet_proj(fft_feat)  # Shape: (N, 256)
        
        # Fuse features using Attention
        fused_features = self.feature_fusion(vit_proj, face_feat, freqnet_feat)  # Shape: (N, 256)
        #Deepseek suggested removing normalization
        # embeddings = nn.functional.normalize(fused_features, p=2, dim=1)
        # Combine features
        
        # Outputs
        main_out = self.classifier(embeddings)
        main_out = main_out.squeeze()

        embeddings = embeddings.squeeze()
        
        return main_out, embeddings

# Training Components
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        BCE_loss = nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss
        return F_loss.mean()

