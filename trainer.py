
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, DistributedSampler
from torch.cuda.amp import GradScaler, autocast
import evaluate
import json
import os
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from model_freqnet import  (
    ArtifactDetector,
    FocalLoss,
    SimpleFusion,
    FeatureProjection
    )
from dataset import RealFakeTripletDataset, RealFakeShuffledDataset
from sklearn.metrics import f1_score
from tqdm import tqdm
# Assume ArtifactDetector, FocalLoss, RealFakeDataset, and Config are correctly imported

class Trainer:
    def __init__(self, model,  config=None, save_dir='v2_output', distributed=True, gpu_id=0):
        """
        Initializes the Trainer with model, device, optimizer, scheduler, loss functions, and data loaders.

        Args:
            model (nn.Module): The neural network model to train.
            gpu_id (int): The GPU ID to use for training.
            config (dict): Configuration dictionary containing hyperparameters.
            save_dir (str): Directory to save models and logs.
            distributed (bool): Flag to enable distributed training.
        """
        torch.autograd.set_detect_anomaly(True)

        self.distributed = distributed
        self.config = config if config is not None else {}
        self.save_dir = save_dir
            
        self.gpu_id = gpu_id
        print (f"Gpu id : {self.gpu_id}")
        self.model = model #already on the currect device
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        self.device= f"cuda:{self.gpu_id}"
        # Wrap the model with DDP if distributed
        if self.distributed==True:
            self.model = DDP(self.model, device_ids=[self.gpu_id], find_unused_parameters=True)
            
       
        # Create save directory only on rank 0
        if self.gpu_id == 0:
            os.makedirs(self.save_dir, exist_ok=True)
        self.num_epochs = self.config.get("num_epochs", 10)
        self.num_epochs_triplet = self.config.get("num_epochs_triplet", 20)

        print (f"Pretraining with triplet loss for {self.num_epochs_triplet} epochs")
        
        # Initialize optimizer and scheduler
        # self.triplet_optimizer = optim.AdamW(self.model.parameters(), lr=self.config.get("lr_triplet", 1e-5))
        #SGD can lead to more generalized results for triplet loss, if hyperparams are finetuned
        model_module = self.model.module if self.distributed else self.model
        self.triplet_params = [p for name, p in model_module.named_parameters() if not name.startswith('classifier')]
        # self.classifier_params = [p for name, p in model_module.named_parameters() if name.startswith('classifier')]
        
        #TODO change back if the model overfits
        self.classifier_params = [p for name, p in model_module.named_parameters() ]
        self.triplet_optimizer = optim.AdamW(self.triplet_params, lr=1e-4)
        self.triplet_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.triplet_optimizer, T_max=self.num_epochs_triplet
        )
        #optimizer for classification: only final layers are finetuned, slower lr
        self.optimizer = optim.AdamW(self.classifier_params, lr=self.config.get("lr", 1e-5))
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=self.num_epochs
        )

        

        # Initialize loss functions
        self.main_criterion = FocalLoss()
        self.triplet_criterion = nn.TripletMarginLoss(margin=config.margin, p=2)


        # Initialize scaler for mixed precision
        #mixed precision can't be used with \fft 
        # self.scaler = GradScaler()

        # Initialize evaluation metric
        self.eval_metric = f1_score

        # Initialize data loaders

        #for triplet loss, no validation
        
        self.train_triplet_loader, self.val_triplet_loader = self.setup_triplet_data_loaders()
        self.train_loader, self.val_loader = self.setup_simple_data_loaders()
        # Initialize history (only on rank 0)
        if self.gpu_id == 0:
            self.history = {
                "train_triplet_loss": [],
                "train_classification_loss": [],
                "val_classification_loss": [],
                "val_f1": []
            }
            self.best_f1 = 0
        self.mode = None
       
        if config.load_checkpoint:
            if 'triplet' in config.load_checkpoint:
                self.mode = 'triplet'   
                self.load_snapshot(config.load_checkpoint)
                #subtract the elapsed epochs
                self.num_epochs_triplet -= self.epoch
            else :
                self.mode = 'main'
                self.load_snapshot(config.load_checkpoint)
                self.num_epochs -= self.epoch
                self.num_epochs_triplet = 0
        else:
            if self.num_epochs_triplet > 0:
                self.mode = 'triplet'
            elif self.num_epochs > 0:
                self.mode = 'main'

                

                    
            #determine where to continue training 

    def setup_simple_data_loaders(self):
        train_set = RealFakeShuffledDataset("/root/workspace/experiments/dataset", self.config, "train")
        val_set = RealFakeShuffledDataset("/root/workspace/experiments/dataset", self.config, "eval")
        if self.distributed:
            train_sampler = DistributedSampler(train_set)
            val_sampler = DistributedSampler(val_set, shuffle=False)
        else:
            train_sampler = None
            val_sampler = None

        train_loader = DataLoader(
            train_set, 
            batch_size=self.config.get("batch_size", 32), 
            shuffle=(train_sampler is None),
            sampler=train_sampler,
            num_workers=0, #self.config.get("num_workers", 4),
            pin_memory=True,
            drop_last=True  # Drop the last incomplete batch

        )
        val_loader = DataLoader(
            val_set, 
            batch_size=self.config.get("batch_size", 32),
            shuffle=False,
            sampler=val_sampler,
            num_workers=0,#self.config.get("num_workers", 4),
            pin_memory=True,
            drop_last=True  # Drop the last incomplete batch

        )
        return train_loader, val_loader

    def setup_triplet_data_loaders(self):
        """
        Sets up the training and validation data loaders with DistributedSampler if in distributed mode.

        Returns:
            tuple: (train_loader, val_loader)
        """
        train_set = RealFakeTripletDataset("/root/workspace/experiments/dataset", self.config, "train")
        val_set = RealFakeTripletDataset("/root/workspace/experiments/dataset", self.config, "eval")

        if self.distributed:
            train_sampler = DistributedSampler(train_set)
            val_sampler = DistributedSampler(val_set, shuffle=False)
        else:
            train_sampler = None
            val_sampler = None

        train_loader = DataLoader(
            train_set, 
            batch_size=self.config.get("batch_size", 32), 
            shuffle=(train_sampler is None),
            sampler=train_sampler,
            num_workers=0, #self.config.get("num_workers", 4),
            pin_memory=True

        )
        val_loader = DataLoader(
            val_set, 
            batch_size=self.config.get("batch_size", 32),
            shuffle=False,
            sampler=val_sampler,
            num_workers=0,#self.config.get("num_workers", 4),
            pin_memory=True

        )
        return train_loader, val_loader

    def pretrain_epoch(self, epoch):
        self.mode = 'triplet'
        self.model.train()
        train_triplet_loss_total = 0.0
        train_classification_loss_total = 0.0
        total_batches = len(self.train_triplet_loader)
        best_loss = 10000;
        if self.distributed:
            self.train_triplet_loader.sampler.set_epoch(epoch)

        for batch in tqdm(self.train_triplet_loader, desc=f"Pre-training Epoch {epoch}"):
            anchor = batch["anchor"].cuda(self.gpu_id, non_blocking=True)
            positive = batch["positive"].cuda(self.gpu_id, non_blocking=True)
            negative = batch["negative"].cuda(self.gpu_id, non_blocking=True)
            
            labels_anchor = torch.zeros(len(anchor), dtype=torch.long).cuda(self.gpu_id)  # Assuming 0 for real
            labels_positive = torch.zeros(len(positive), dtype=torch.long).cuda(self.gpu_id)  # Same class as anchor
            labels_negative = torch.ones(len(negative), dtype=torch.long).cuda(self.gpu_id)   # Different class

            combined = torch.cat([anchor, positive, negative], dim=0)  # Shape: (3B, C, H, W)
            # Forward pass
            _, embeddings = self.model(combined)  # embeddings shape: (3B, 256)
            
            # Split embeddings
            embeddings_anchor, embeddings_positive, embeddings_negative = embeddings.chunk(3, dim=0)
            
            
            # Triplet Loss
            total_loss = self.triplet_criterion(embeddings_anchor, embeddings_positive, embeddings_negative)

            # Backward and Optimize
            self.triplet_optimizer.zero_grad()
            total_loss.backward()
            
            torch.nn.utils.clip_grad_norm_(self.triplet_params, max_norm=1.0)
            
            self.triplet_optimizer.step()
            self.triplet_scheduler.step()

            # Accumulate losses
            train_triplet_loss_total = train_triplet_loss_total + total_loss.item()

        # Calculate average losses
        avg_triplet_loss = train_triplet_loss_total / total_batches

        if self.gpu_id == 0:
            print(f"Epoch {epoch}: Train Triplet Loss: {avg_triplet_loss:.4f}")
            self.history["train_triplet_loss"].append(avg_triplet_loss)
            if avg_triplet_loss < best_loss:
                self.save_snapshot(epoch, best=True)
                best_loss = avg_triplet_loss
                
    def train_epoch(self, epoch):
        """
        Runs a full training epoch using triplet loss. Tunes the model's embeddings before the main
        training loop. 
        Args:
            epoch (int): Current epoch number.
        """
        self.model.train()
        self.mode = 'main'
        train_classification_loss_total = 0.0
        total_batches = len(self.train_loader)

        if self.distributed:
            self.train_loader.sampler.set_epoch(epoch)

        for batch in tqdm(self.train_loader, desc=f"Training Epoch {epoch}"):
            imgs, labels = batch
            imgs = imgs.cuda(self.gpu_id)
            labels = labels.cuda(self.gpu_id)
            # Forward pass: compute embeddings and logits
            logits, _ = self.model(imgs)
            
            # # Classification Loss
            # # Concatenate all logits and labels
            total_loss = self.main_criterion(logits.float(), labels.float())


            # Backward and Optimize
            self.optimizer.zero_grad()
            total_loss.backward()
            
            torch.nn.utils.clip_grad_norm_(self.classifier_params, max_norm=1.0)
            
            self.optimizer.step()
            self.scheduler.step()
            # Accumulate losses
            train_classification_loss_total = train_classification_loss_total + total_loss.item()

        # Calculate average losses
        avg_classification_loss = train_classification_loss_total / total_batches
        # avg_classification_loss = train_classification_loss_total / total_batches

        if self.gpu_id == 0:
            print(f"Epoch {epoch}: Train Classification Loss: {avg_classification_loss:.4f}")
            self.history["train_classification_loss"].append(avg_classification_loss)
            # self.history["train_classification_loss"].append(avg_classification_loss)
            

    def eval_epoch(self, epoch):
        """
        Runs a full evaluation epoch using classification metrics.
        Args:
            epoch (int): Current epoch number.
        """
        self.model.eval()
        self.mode = 'main'
        val_classification_loss_total = 0.0
        all_predictions = []
        all_labels = []
        total_batches = len(self.val_loader)
        total_val_loss = 0
        val_f1_total = 0
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc=f"Validation Epoch {epoch}"):
                imgs, labels = batch
                imgs = imgs.cuda(self.gpu_id)
                labels = labels.cuda(self.gpu_id)
                # Forward pass: compute embeddings and logits
                logits, _ = self.model(imgs)
                val_loss = self.main_criterion(logits.float(), labels.float())
                total_val_loss += val_loss.item()
                # Compute F1 score
                probs = torch.sigmoid(logits)
                predictions = (probs > 0.5).float()
                print (f"Predictions: {predictions.cpu()}, Labels: {labels.cpu()}")
                val_f1 = self.eval_metric(labels.cpu(), predictions.cpu())
                val_f1_total += val_f1
                
                

        # Calculate average validation loss
        avg_val_classification_loss = total_val_loss / total_batches
        avg_f1 = val_f1_total / total_batches
        if self.gpu_id == 0:
            print(f"Epoch {epoch}: Val Classification Loss: {avg_val_classification_loss:.4f}, F1 Score: {avg_f1:.4f}")
            self.history["val_classification_loss"].append(avg_val_classification_loss)
            self.history["val_f1"].append(avg_f1)

            # Save the best model based on F1 Score
            if avg_f1 > self.best_f1:
                self.best_f1 = avg_f1
                self.save_snapshot(epoch, best=True)

    def save_snapshot(self, epoch, best=False):
        """
        Saves the current state of the model and optimizer.

        Args:
            epoch (int): Current epoch number.
            best (bool): Whether this snapshot is the best model so far.
        """
        if self.mode == 'main':
            state = {
                'epoch': epoch,
                'model_state_dict': self.model.module.state_dict() if self.distributed else self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict(),
                'history': self.history,
                'best_f1': self.best_f1
            }
            filename = f"best_model.pth" if best else f"model_epoch_{epoch}.pth"
            filepath = os.path.join(self.save_dir, filename)
            torch.save(state, filepath)
            print(f"Saved {'best ' if best else ''}model at epoch {epoch} to {filepath}")
        elif self.mode == 'triplet':
            state = {
                'epoch': epoch,
                'model_state_dict': self.model.module.state_dict() if self.distributed else self.model.state_dict(),
                'optimizer_state_dict': self.triplet_optimizer.state_dict(),
                'scheduler_state_dict': self.triplet_scheduler.state_dict(),
                'history': self.history,
            }
            filename = f"best_model_triplet.pth" if best else f"model_epoch_{epoch}.pth"
            filepath = os.path.join(self.save_dir, filename)
            torch.save(state, filepath)
            print(f"Saved {'best ' if best else ''} triplet model at epoch {epoch} to {filepath}")
    


    def load_snapshot(self, filepath):
        """
        Loads the model and optimizer state from a snapshot.
        Args:
            filepath (str): Path to the snapshot file.
        """
        if not os.path.isfile(filepath):
            raise FileNotFoundError(f"No snapshot found at {filepath}")
        
        # Load the checkpoint
        checkpoint = torch.load(filepath, map_location=self.device)
        
        # Load model state_dict
        if self.distributed:
            self.model.module.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load history
        self.history = checkpoint.get('history', {})
        
        # Handle different modes
        if self.mode == 'main':
            self.epoch = checkpoint.get('epoch', 0)
    
            # Load optimizer state_dict
            if 'optimizer_state_dict' in checkpoint:
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            else:
                print("Warning: 'optimizer_state_dict' not found in checkpoint for 'main' mode.")
            
            # Load scheduler state_dict
            if 'scheduler_state_dict' in checkpoint:
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            else:
                print("Warning: 'scheduler_state_dict' not found in checkpoint for 'main' mode.")
            
            # Load best_f1
            self.best_f1 = checkpoint.get('best_f1', 0.0)
        
        elif self.mode == 'triplet':
            self.epoch = checkpoint.get('epoch', 0)
    
            # Load triplet optimizer state_dict
            if 'optimizer_state_dict' in checkpoint:
                self.triplet_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            else:
                print("Warning: 'optimizer_state_dict' not found in checkpoint for 'triplet' mode.")
            # Load triplet scheduler state_dict
            if 'scheduler_state_dict' in checkpoint:
                self.triplet_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            else:
                print("Warning: 'scheduler_state_dict' not found in checkpoint for 'triplet' mode.")
        else:
            raise ValueError(f"Unknown mode: {self.mode}")
        
        # Print loaded message
        if self.gpu_id == 0:
            print(f"Loaded snapshot from {filepath} in '{self.mode}' mode at epoch {self.epoch}.")

    def train_loop(self):
        """
        Manages the overall training and evaluation loop across all epochs.
        """
        # Log the model architecture (only on rank 0)
        if self.gpu_id == 0:
            model_arch_path = os.path.join(self.save_dir, "model_architecture.txt")
            with open(model_arch_path, "w") as f:
                f.write(str(self.model.module) if self.distributed else str(self.model))
        #pretraining embeddings
        for epoch in range(1, self.num_epochs_triplet + 1):
            self.pretrain_epoch(epoch)
            

        #training classifier
        for epoch in range(1, self.num_epochs + 1):
            self.train_epoch(epoch)
            self.eval_epoch(epoch)
            

        if self.gpu_id == 0:
            # Save the entire history
            history_path = os.path.join(self.save_dir, "training_history.json")
            with open(history_path, "w") as f:
                json.dump(self.history, f, indent=4)
            print(f"Training complete. History saved to {history_path}")
        
