from model_freqnet import ArtifactDetector, RealFakeDataset

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
    RealFakeDataset,
    FocalLoss,
    AttentionFusion,
    FeatureProjection
    )
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
        self.distributed = distributed
        self.config = config if config is not None else {}
        self.save_dir = save_dir
            
        self.gpu_id = gpu_id
        print (f"Gpu id : {self.gpu_id}")
        self.model = model #already on the currect device
        self.device= f"cuda:{self.gpu_id}"
        # Wrap the model with DDP if distributed
        if self.distributed==True:
            self.model = DDP(self.model, device_ids=[self.gpu_id], find_unused_parameters=True)

        # Create save directory only on rank 0
        if self.gpu_id == 0:
            os.makedirs(self.save_dir, exist_ok=True)

        # Initialize optimizer and scheduler
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.config.get("lr", 1e-5))
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=self.config.get("num_epochs", 10)
        )

        # Initialize loss functions
        self.main_criterion = FocalLoss()

        # Initialize scaler for mixed precision
        #mixed precision can't be used with \fft 
        # self.scaler = GradScaler()

        # Initialize evaluation metric
        self.eval_metric = evaluate.load('f1')

        # Initialize data loaders
        self.train_loader, self.val_loader = self.setup_data_loaders()

        # Initialize history (only on rank 0)
        if self.gpu_id == 0:
            self.history = {
                "train_loss": [],
                "val_loss": [],
                "val_f1": []
            }
            self.best_f1 = 0

    def setup_data_loaders(self):
        """
        Sets up the training and validation data loaders with DistributedSampler if in distributed mode.

        Returns:
            tuple: (train_loader, val_loader)
        """
        train_set = RealFakeDataset("/root/workspace/dataset", self.config, "train")
        val_set = RealFakeDataset("/root/workspace/dataset", self.config, "eval")

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

    def train_epoch(self, epoch):
        """
        Runs a full training epoch.

        Args:
            epoch (int): Current epoch number.
        """
        self.model.train()

        train_total_loss = 0.0
        total_batches = len(self.train_loader)

        if self.distributed:
            self.train_loader.sampler.set_epoch(epoch)

        for batch in tqdm(self.train_loader):
            
            real_rgb = batch["real"].cuda(self.gpu_id, non_blocking=True)
            fake_rgb = batch["fake"].cuda(self.gpu_id, non_blocking=True)
            rgb = torch.cat([real_rgb, fake_rgb])
            labels = torch.cat([
                torch.zeros(len(real_rgb)),
                torch.ones(len(fake_rgb))
            ]).cuda(self.gpu_id, non_blocking=True)

            # Forward pass with autocast for mixed precision
            self.optimizer.zero_grad()

            main_out = self.model(rgb)
            loss_main = self.main_criterion(main_out, labels)

            # Backward pass and optimizer step
            loss_main.backward()
            self.optimizer.step()
            self.scheduler.step()

            train_total_loss += loss_main.item()

            # self.scaler.scale(total_loss).backward()
            # self.scaler.step(self.optimizer)
            # self.scaler.update()


        # Calculate average losses
        avg_loss_main = train_total_loss / total_batches

        if self.gpu_id == 0:
            print(f"Epoch {epoch}: Train Main Loss: {avg_loss_main:.4f}")
            self.history["train_loss"].append(avg_loss_main)

    def eval_epoch(self, epoch):
        """
        Runs a full evaluation epoch.

        Args:
            epoch (int): Current epoch number.
        """
        self.model.eval()
        val_loss_total = 0.0
        val_f1_total = 0.0
        total_batches = len(self.val_loader)

        with torch.no_grad():
            for batch in self.val_loader:
                real_rgb = batch["real"].cuda(self.gpu_id, non_blocking=True)
                fake_rgb = batch["fake"].cuda(self.gpu_id, non_blocking=True)
                rgb = torch.cat([real_rgb, fake_rgb])
                labels = torch.cat([
                    torch.zeros(len(real_rgb)),
                    torch.ones(len(fake_rgb))
                ]).cuda(self.gpu_id, non_blocking=True)

                # Forward pass
                main_out = self.model(rgb)
                loss_main = self.main_criterion(main_out, labels)
                val_loss_total += loss_main.item()

                # Compute F1 score
                probs = torch.sigmoid(main_out)
                predictions = (probs > 0.5).float()
                val_f1 = self.eval_metric.compute(predictions=predictions.cpu(), references=labels.cpu())['f1']
                val_f1_total += val_f1

        # Calculate average validation loss and F1 score
        avg_val_loss = val_loss_total / total_batches
        avg_val_f1 = val_f1_total / total_batches

        if self.gpu_id == 0:
            print(f"Epoch {epoch}: Val Loss: {avg_val_loss:.4f}, F1 Score: {avg_val_f1:.4f}")
            self.history["val_loss"].append(avg_val_loss)
            self.history["val_f1"].append(avg_val_f1)

            # Save the best model
            if avg_val_f1 > self.best_f1:
                self.best_f1 = avg_val_f1
                self.save_snapshot(epoch, best=True)

    def save_snapshot(self, epoch, best=False):
        """
        Saves the current state of the model and optimizer.

        Args:
            epoch (int): Current epoch number.
            best (bool): Whether this snapshot is the best model so far.
        """
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

    def load_snapshot(self, filepath):
        """
        Loads the model and optimizer state from a snapshot.

        Args:
            filepath (str): Path to the snapshot file.
        """
        if not os.path.isfile(filepath):
            raise FileNotFoundError(f"No snapshot found at {filepath}")

        checkpoint = torch.load(filepath, map_location=self.device)
        if self.distributed:
            self.model.module.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.history = checkpoint.get('history', {})
        self.best_f1 = checkpoint.get('best_f1', 0.0)
        if self.gpu_id == 0:
            print(f"Loaded snapshot from {filepath}")

    def train_loop(self):
        """
        Manages the overall training and evaluation loop across all epochs.
        """
        # Log the model architecture (only on rank 0)
        if self.gpu_id == 0:
            model_arch_path = os.path.join(self.save_dir, "model_architecture.txt")
            with open(model_arch_path, "w") as f:
                f.write(str(self.model.module) if self.distributed else str(self.model))

        num_epochs = self.config.get("num_epochs", 10)
        for epoch in range(1, num_epochs + 1):
            self.train_epoch(epoch)
            self.eval_epoch(epoch)
            self.scheduler.step()

        if self.gpu_id == 0:
            # Save the entire history
            history_path = os.path.join(self.save_dir, "training_history.json")
            with open(history_path, "w") as f:
                json.dump(self.history, f, indent=4)
            print(f"Training complete. History saved to {history_path}")
        