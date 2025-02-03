import torch
import argparse
import json
from trainer import Trainer
from model_freqnet import  (
    ArtifactDetector,
    RealFakeDataset,
    FocalLoss,
    AttentionFusion,
    FeatureProjection
)
import torch.distributed as dist

import os
    # Replace 'your_module' with actual module name

class Config:
    def __init__(self):
        self.img_size = 224
        self.batch_size =4
        self.lr = 1e-5
        self.num_epochs = 10
        self.aux_lambda = 0.1
        # self.vit_model = "WinKawaks/vit-tiny-patch16-224"
        self.vit_model = 'prithivMLmods/Deep-Fake-Detector-v2-Model'
        self.save_dir = "freqnet_mtcnn_vit_aug_res"
        self.debug = False
        self.log_artifacts = True  # Save models and plots locally
        self.common_dim = 256      # Dimension of feature projection for all branches
        self.freqnet_dim = 512
        self.num_heads = 8
        self.num_layers = 2
        self.dropout=0.1
        self.freqnet_weights = 'freqnet_weights.pth'

    def get(self, key, default=None):
        """
        Retrieve an attribute by name with an optional default value.
        
        Args:
            key (str): The name of the attribute to retrieve.
            default: The value to return if the attribute is not found.

        Returns:
            The value of the attribute if it exists; otherwise, the default value.
        """
        return getattr(self, key, default)
    
    # def to_dict(self):
    #     """
    #     Convert all instance attributes to a dictionary.
        
    #     Returns:
    #         dict: A dictionary containing all configuration parameters.
    #     """
    #     return {attr: value for attr, value in self.__dict__.items() if not callable(value) and not attr

def main():
    """ https://github.com/pytorch/elastic/blob/master/examples/imagenet/main.py"""
    assert torch.cuda.is_available()
    parser = argparse.ArgumentParser(description="Distributed Training with Trainer Class")
    parser.add_argument('--save_dir', type=str, default='freqnet_mtcnn_vit_aug_res', help='Directory to save models and logs.')
    args = parser.parse_args()
    
   

    # # Load configuration TODO
    # with open(args.config, 'r') as f:
    #     config = json.load(f)
    config = Config()
    # Initialize the model

    # Initialize the Trainer with distributed flag set based on environment
    distributed = True if "LOCAL_RANK" in os.environ else False

    if distributed:
        dist.init_process_group(backend='nccl')
        gpu_id = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(gpu_id)
    else:
        gpu_id = 0
    model = ArtifactDetector(config, gpu_id).cuda(gpu_id)  # Replace with your actual model initialization

    
    trainer = Trainer(
        model=model,  
        config=config, 
        save_dir=args.save_dir, 
        distributed=distributed,
        gpu_id=gpu_id
    )

    # Start the training loop
    trainer.train_loop()
    if distributed:
        dist.destroy_process_group()

if __name__ == "__main__":
    main()