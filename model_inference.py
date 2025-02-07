import torch
import torch.nn.functional as F
from torchvision import transforms
import cv2
import numpy as np
from PIL import Image
from model_freqnet import ArtifactDetector
from sklearn.metrics import classification_report
import pandas as pd
import sys
import os
import argparse
from run_model_training import Config
from torch.utils.data import Dataset, DataLoader
import pickle as pkl
class ArtifactInferenceDataset(Dataset):
    def __init__(self, image_paths):
        self.image_paths = image_paths
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def load_image(self, path):
        """Custom image loader with error handling"""
        
        img = Image.open(path).convert('RGB')
        return self.transform(img)
    def __len__(self):
        return len(self.image_paths)
        
    
    def __getitem__(self, idx):
        path = self.image_paths[idx]
        img = self.load_image(path)
        
        # Face detection
        # has_face = self.detect_face(img)
        return {'path':path,  'img':img}
        

class ArtifactInference:
    def __init__(self, model_path, device="cuda:0" if torch.cuda.is_available() else "cpu"):
        self.device = device
        self.model = self.load_model(model_path)
        
        # Inference-specific transforms (match model's training expectations)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def load_model(self, path):
        model = ArtifactDetector(Config(), 0)
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        return model.to(self.device).eval()

    def preprocess(self, image_path):
        # Load and transform image
        img = Image.open(image_path).convert('RGB')
        img_tensor = self.transform(img).unsqueeze(0)  # Add batch dim
        
       
        
        # Detect faces
        img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        has_face = 1 if len(self.face_detector.detect_faces(img_cv)) > 0 else 0
        
        return img_tensor.to(self.device), fft_tensor.to(self.device), torch.tensor([has_face], device=self.device)

   

    def batch_predict(self, image_paths, batch_size=32):
        dataset = ArtifactInferenceDataset(image_paths)  # Custom dataset
        loader = DataLoader(dataset, batch_size=batch_size)
        
        results = []
        for batch in loader:
            with torch.no_grad():
                path, img = batch['path'], batch['img']
                main_outs, _ = self.model(img.to(self.device))
                probs = torch.sigmoid(main_outs)
                
            # batch_results = [{
            #     "prediction": (probs > 0.5).float(),
            #     # "confidence": p.item() if p > 0.5 else 1 - p.item()
            #     # "has_face": bool(hf.item())
            # } for p in zip(probs)]
            batch_results = (probs > 0.5).float().cpu()
            results.extend(batch_results)
        
        return results

    def batch_eval(self, image_root):
        real_image_paths = os.listdir(os.path.join(image_root, 'real'))
        real_image_paths = [os.path.join(image_root, 'real', i) for i in real_image_paths]
        fake_image_paths = os.listdir(os.path.join(image_root, 'fake'))
        fake_image_paths = [os.path.join(image_root, 'fake',i) for i in fake_image_paths]
        # gt_labels = np.concatenate([np.zeros(len(real_image_paths)),
        #                     np.ones(len(fake_image_paths))])
        # image_paths = real_image_paths
        # image_paths.extend(fake_image_paths)
        gt_labels = np.concatenate([
                            np.ones(len(fake_image_paths)),
                                   np.zeros(len(real_image_paths))])
        image_paths =  fake_image_paths
        image_paths.extend(real_image_paths)
        predictions = self.batch_predict(image_paths)
        # predictions = [0 if i['prediction']=='real' else 1 for i in prediction_dicts]
        return classification_report(gt_labels, predictions, target_names=['real', 'fake']), gt_labels, predictions
        

# Usage Example
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Distributed Training with Trainer Class")
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to  model checkpoint.')
    parser.add_argument('--img_root', type=str, default=None, help='Path to  eval image root path .')

    args = parser.parse_args()

    detector = ArtifactInference(args.checkpoint)
    if not os.path.exists(args.img_root):
        print (f"No dataset found at {args.img_root} path!")
    else:
        report, gt, preds = detector.batch_eval(args.img_root)
        if isinstance(report, dict):
            report = pd.DataFrame(report)
            report.to_csv('evaluation_report.csv')
        elif isinstance(report, str):
            print (report)
            
        with open('gt.pkl', 'wb') as f:
            pkl.dump(gt, f)
        with open('preds.pkl', 'wb') as f:
            pkl.dump(pred, f)
    # result = detector.predict("test_image.jpg")
    # print(f"Prediction: {result['prediction']} (Confidence: {result['confidence']:.2%})")
    # print(f"Contains faces: {result['has_face']}")
