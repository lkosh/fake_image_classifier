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

from dataset import RealFakeDataset

#layer classes 
# class AttentionFusion(nn.Module):
#     def __init__(self, feature_dim=256, num_heads=8, num_layers=2, dropout=0.1):
#         super(AttentionFusion, self).__init__()
#         encoder_layers = TransformerEncoderLayer(d_model=feature_dim, nhead=num_heads, dropout=dropout)
#         self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers=num_layers)
#         self.fc = nn.Linear(feature_dim, feature_dim)
#         self.relu = nn.ReLU()

#     def forward(self, vit, facenet, freqnet):
#         # Combine features into a sequence
#         x = torch.stack([vit, facenet, freqnet], dim=1)  # Shape: (N, 3, D)
#         x = x.permute(1, 0, 2)  # Transformer expects (Seq_len, Batch, D)

#         # Apply Transformer Encoder
#         x = self.transformer_encoder(x)  # Shape: (3, N, D)

#         # Aggregate the sequence (e.g., take the mean)
#         x = x.mean(dim=0)  # Shape: (N, D)

#         # Optional: Further processing
#         x = self.fc(x)
#         x = self.relu(x)
#         return x

class AttentionFusion(nn.Module):
    def __init__(self, feature_dim=256, num_heads=8, num_layers=2, dropout=0.1):
        super(AttentionFusion, self).__init__()
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
        self.attention_fusion = AttentionFusion(feature_dim=common_dim)
        self.init_weights()

    def init_weights(self):
        #initialize freqnet:
        state_dict = torch.load(self.config.freqnet_weights, map_location=f"cuda:{self.gpu_id}")
        self.freqnet.load_state_dict(state_dict, strict=False)
        #pytorch initializes layers with uniform distr.
        #optional TODO: add more advanced weight init
        # for layer in [self.face_aux, self.face_net.modules()]:
        #     nn.init.xavier_uniform_(layer.weight)
        #     nn.init.zeros_(layer.bias)
        
    def forward(self, rgb):
        batch_dim = rgb.size(0)#
        # RGB branch
        vit_out = self.vit(rgb).last_hidden_state[:, 0]
        # vit_feat = self.vit_fc(vit_out)
        
        # Frequency branch
        fft_feat = self.freqnet(rgb).squeeze() #512 channels

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
        # facenet_proj = self.facenet_proj(facenet)  # Shape: (N, 256)
        freqnet_proj = self.freqnet_proj(fft_feat)  # Shape: (N, 256)
        
        # Fuse features using Attention
        fused_features = self.attention_fusion(vit_proj, face_feat, freqnet_proj)  # Shape: (N, 256)
        
        # Combine features
        
        # Outputs
        main_out = self.classifier(fused_features)
        #2 losses overcomplicate things, we're not trying to train a face detector!
        # aux_out = self.face_aux(face_feat) 
        
        return main_out.squeeze()

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

# def train_model(config):
#     scaler = GradScaler()
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     eval_metric = evaluate.load('f1')
    
#     history = {
#         "train_loss_main": [],
#         "train_loss_aux":[],
#         "val_loss": [],
#         "val_f1": []
#     }

        
#     # Log the model architecture


    
#     # Initialize model
#     model = ArtifactDetector().to(device)
    
#     with open(f"{Config.save_dir}/model_architecture.txt", "w") as f:
#         f.write(str(model))
#     # with open(f"{Config.save_dir}/Config.json", "w") as f:
#     #     json.dump(Config.to_dict(), f)
    
        
#     optimizer = optim.AdamW(model.parameters(), lr=config["lr"])
#     scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Config.num_epochs)
    
#     # Losses
#     main_criterion = FocalLoss()
#     aux_criterion = nn.BCEWithLogitsLoss()
    
#     # Data
#     train_set = RealFakeDataset("/root/research/dataset", "train")
#     val_set = RealFakeDataset("/root/research/dataset", "eval")
    
#     train_loader = DataLoader(train_set, batch_size=config["batch_size"], shuffle=True)
#     val_loader = DataLoader(val_set, batch_size=config["batch_size"])

#     best_f1 = 0
#     # Training loop
#     train_loss_main = 0
#     train_loss_aux= 0
#     for epoch in range(Config.num_epochs):
#         model.train()
#         for batch in train_loader:
#             real_rgb = batch["real"]
#             fake_rgb = batch["fake"]
            
#             # Combine real and fake samples
#             rgb = torch.cat([real_rgb, fake_rgb]).to(device)
#             labels = torch.cat([
#                 torch.zeros(len(real_rgb)),
#                 torch.ones(len(fake_rgb))
#             ]).to(device)
#             has_face = torch.cat([batch["has_face"], batch["has_face"]]).to(device)
            
#             # Forward
#             main_out, aux_out = model(rgb)
#             loss_main = main_criterion(main_out, labels)
#             loss_aux = aux_criterion(aux_out, has_face)
#             total_loss = loss_main + Config.aux_lambda * loss_aux
#             train_loss_main += loss_main.item()
#             train_loss_aux += loss_aux.item()
            
#             # Backward
#             optimizer.zero_grad()
#             total_loss.backward()
#             optimizer.step()
#         print (f"Epoch {epoch} main train loss: {train_loss_main / len(train_loader)}, aux train loss: {train_loss_aux/len(train_loader)}")
#         history["train_loss_main"].append(train_loss_main / len(train_loader))
#         history['train_loss_aux'].append(train_loss_aux/len(train_loader))

#         # Validation
#         model.eval()
#         val_loss_main = 0
#         val_loss_total =0
#         val_f1_total = 0
#         with torch.no_grad():
#             for batch in val_loader:
#                 real_rgb = batch["real"]
#                 fake_rgb= batch["fake"]
                
#                 # Combine real and fake samples
#                 rgb = torch.cat([real_rgb, fake_rgb]).to(device)
#                 labels = torch.cat([
#                     torch.zeros(len(real_rgb)),
#                     torch.ones(len(fake_rgb))
#                 ]).to(device)
#                 has_face = torch.cat([batch["has_face"], batch["has_face"]]).to(device)
                
#                 # Forward
#                 main_out, aux_out = model(rgb)
#                 probs = torch.sigmoid(main_out) #we got 1 output neuron 
#                 predictions = (probs > 0.5).float()
                
#                 val_f1 = eval_metric.compute(predictions=predictions, references=labels)['f1']
#                 val_loss_main = main_criterion(main_out, labels)
#                 val_loss_total += val_loss_main.item()
#                 val_f1_total += val_f1
#                 # val_loss_aux = aux_criterion(aux_out, has_face)
#                 # total_loss = loss_main + 0.3 * loss_aux
#             print (f"Epoch {epoch} eval loss: {val_loss_main/ len(val_loader)}, f1: {val_f1_total/len(val_loader)}")
#             history["val_loss"].append(val_loss_total)
#             history["val_f1"].append(val_f1_total)


#         # Log metrics
       
        
#         if val_f1 > best_f1:
#                 best_f1 = val_f1
#                 torch.save(model.state_dict(), f"{Config.save_dir}/best_model.pth")
        
#         # Final logging
        
        
#         scheduler.step()
#     return history

#     # tune.report(val_f1=val_f1)


# # Main Execution
# if __name__ == "__main__":
#     cfg = {
#         'lr':1e-6,
#         'batch_size':1
#     }
        
#     history = train_model(cfg)
#     with open(f"{Config.save_dir}/train_logs.pkl", "wb") as f:
#         pkl.dump(history, f)
#     # tune_model() dont need tuning yet
    