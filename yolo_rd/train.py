"""
Training script for YOLO-RD with Roboflow dataset integration
"""

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
from pathlib import Path
import yaml
from tqdm import tqdm
import warnings

try:
    from roboflow import Roboflow
    ROBOFLOW_AVAILABLE = True
except ImportError:
    ROBOFLOW_AVAILABLE = False
    warnings.warn("Roboflow not available. Install with: pip install roboflow")

try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False
    warnings.warn("Ultralytics not available. Install with: pip install ultralytics")

from yolo_rd.models.yolo_rd import create_yolo_rd_model
from yolo_rd.models.config import yolo_rd_simple_config


class RoboflowDatasetLoader:
    """
    Loader for Roboflow datasets
    Dataset: https://universe.roboflow.com/road-damage-detection-n2xkq/crack-and-pothole-bftyl
    """
    def __init__(self, api_key, workspace="road-damage-detection-n2xkq", 
                 project="crack-and-pothole-bftyl", version=1):
        self.api_key = api_key
        self.workspace = workspace
        self.project = project
        self.version = version
        self.dataset = None
    
    def download_dataset(self, format="yolov8", location="./datasets"):
        """
        Download dataset from Roboflow
        
        Args:
            format: Dataset format (yolov8, coco, etc.)
            location: Download location
        """
        if not ROBOFLOW_AVAILABLE:
            raise RuntimeError("Roboflow package not installed")
        
        rf = Roboflow(api_key=self.api_key)
        project = rf.workspace(self.workspace).project(self.project)
        dataset = project.version(self.version).download(format, location=location)
        
        self.dataset = dataset
        return dataset.location
    
    def get_data_yaml_path(self):
        """Get path to data.yaml file"""
        if self.dataset is None:
            raise RuntimeError("Dataset not downloaded yet")
        return Path(self.dataset.location) / "data.yaml"


class YOLORDTrainer:
    """
    Trainer for YOLO-RD model
    """
    def __init__(self, model, config, device='cuda'):
        self.model = model.to(device)
        self.config = config
        self.device = device
        
        # Setup optimizer
        train_config = config.get('train', {})
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=train_config.get('lr0', 0.001),
            weight_decay=train_config.get('weight_decay', 0.0005)
        )
        
        # Setup learning rate scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=train_config.get('epochs', 100),
            eta_min=train_config.get('lrf', 0.01) * train_config.get('lr0', 0.001)
        )
        
        self.epoch = 0
        self.best_metric = 0.0
    
    def train_epoch(self, dataloader):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        loss_items = {'cls': 0.0, 'box': 0.0, 'dfl': 0.0}
        
        pbar = tqdm(dataloader, desc=f'Epoch {self.epoch}')
        for batch_idx, (images, targets) in enumerate(pbar):
            images = images.to(self.device)
            # Move targets to device
            if isinstance(targets, dict):
                targets = {k: v.to(self.device) if torch.is_tensor(v) else v 
                          for k, v in targets.items()}
            
            # Forward pass
            loss, loss_dict = self.model(images, targets)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Track losses
            total_loss += loss.item()
            for k in loss_items.keys():
                if k in loss_dict:
                    loss_items[k] += loss_dict[k].item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': loss.item(),
                'cls': loss_dict.get('cls', 0.0).item() if torch.is_tensor(loss_dict.get('cls', 0.0)) else 0.0,
                'box': loss_dict.get('box', 0.0).item() if torch.is_tensor(loss_dict.get('box', 0.0)) else 0.0
            })
        
        # Average losses
        avg_loss = total_loss / len(dataloader)
        avg_loss_items = {k: v / len(dataloader) for k, v in loss_items.items()}
        
        return avg_loss, avg_loss_items
    
    def validate(self, dataloader):
        """Validate model"""
        self.model.eval()
        # Simplified validation - implement proper metrics
        return 0.0
    
    def train(self, train_loader, val_loader, epochs, save_dir='./runs/train'):
        """
        Full training loop
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Number of epochs to train
            save_dir: Directory to save checkpoints
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        for epoch in range(epochs):
            self.epoch = epoch
            
            # Train
            train_loss, train_loss_items = self.train_epoch(train_loader)
            
            # Validate
            val_metric = self.validate(val_loader)
            
            # Update learning rate
            self.scheduler.step()
            
            # Save checkpoint
            if val_metric > self.best_metric:
                self.best_metric = val_metric
                self.save_checkpoint(save_dir / 'best.pt')
            
            self.save_checkpoint(save_dir / 'last.pt')
            
            print(f"\nEpoch {epoch}/{epochs}")
            print(f"Train Loss: {train_loss:.4f}")
            print(f"  - Classification: {train_loss_items['cls']:.4f}")
            print(f"  - Localization: {train_loss_items['box']:.4f}")
            print(f"  - DFL: {train_loss_items['dfl']:.4f}")
    
    def save_checkpoint(self, path):
        """Save model checkpoint"""
        torch.save({
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_metric': self.best_metric,
        }, path)
    
    def load_checkpoint(self, path):
        """Load model checkpoint"""
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.epoch = checkpoint['epoch']
        self.best_metric = checkpoint['best_metric']


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train YOLO-RD model')
    parser.add_argument('--data', type=str, help='Path to data.yaml or Roboflow API key')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    parser.add_argument('--img-size', type=int, default=640, help='Image size')
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda/cpu)')
    parser.add_argument('--workers', type=int, default=4, help='Number of workers')
    parser.add_argument('--project', type=str, default='runs/train', help='Save directory')
    parser.add_argument('--name', type=str, default='exp', help='Experiment name')
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    
    # Roboflow specific
    parser.add_argument('--roboflow-api-key', type=str, help='Roboflow API key')
    parser.add_argument('--roboflow-workspace', type=str, 
                       default='road-damage-detection-n2xkq', help='Roboflow workspace')
    parser.add_argument('--roboflow-project', type=str, 
                       default='crack-and-pothole-bftyl', help='Roboflow project')
    
    return parser.parse_args()


def main():
    """Main training function"""
    args = parse_args()
    
    print("=" * 50)
    print("YOLO-RD Training")
    print("=" * 50)
    
    # Create model
    print("\nCreating YOLO-RD model...")
    model = create_yolo_rd_model(num_classes=2)
    model_info = model.get_model_info()
    print(f"Parameters: {model_info['parameters_M']:.2f}M")
    print(f"Target: {model_info['target_parameters_M']}M parameters, "
          f"{model_info['target_gflops']} GFLOPs")
    
    # Setup data
    if args.roboflow_api_key:
        print("\nDownloading dataset from Roboflow...")
        loader = RoboflowDatasetLoader(
            api_key=args.roboflow_api_key,
            workspace=args.roboflow_workspace,
            project=args.roboflow_project
        )
        dataset_path = loader.download_dataset(location="./datasets")
        data_yaml = loader.get_data_yaml_path()
        print(f"Dataset downloaded to: {dataset_path}")
    elif args.data:
        data_yaml = args.data
    else:
        raise ValueError("Either --data or --roboflow-api-key must be provided")
    
    # Note: Actual data loading would require integration with YOLOv8 data loaders
    # or custom implementation. This is simplified for demonstration.
    print("\nNote: For full training, integrate with YOLOv8 data loaders or implement custom data loading")
    
    # Create trainer
    trainer = YOLORDTrainer(
        model=model,
        config=yolo_rd_simple_config,
        device=args.device
    )
    
    # Load checkpoint if resuming
    if args.resume:
        print(f"\nResuming from checkpoint: {args.resume}")
        trainer.load_checkpoint(args.resume)
    
    print("\nModel ready for training!")
    print("To start training, implement data loaders and call:")
    print("  trainer.train(train_loader, val_loader, epochs=args.epochs)")


if __name__ == '__main__':
    main()
