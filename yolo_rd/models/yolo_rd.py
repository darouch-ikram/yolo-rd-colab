"""
YOLO-RD Model Implementation
Basé sur YOLOv8s avec modules personnalisés (CSAF, LGECA, LFC)
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
import warnings

try:
    from ultralytics import YOLO
    from ultralytics.nn.modules import Conv, C2f, SPPF, Detect, Concat
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False
    warnings.warn("Ultralytics not available. Install with: pip install ultralytics")

from ..modules import CSAF, LGECA, SR_WBCE_Loss
from .config import yolo_rd_simple_config


class LFC(nn.Module):
    """
    Layer-wise Feature Compression
    Réduit les canaux (512→256) pour optimiser les paramètres
    """
    def __init__(self, in_channels, out_channels):
        super(LFC, self).__init__()
        self.compress = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True)
        )
    
    def forward(self, x):
        return self.compress(x)


class YOLORDBackbone(nn.Module):
    """
    YOLO-RD Backbone with custom modules
    """
    def __init__(self, in_channels=3, base_channels=64):
        super(YOLORDBackbone, self).__init__()
        
        # Layer 0: CSAF (replaces first Conv)
        self.layer0 = CSAF(in_channels, base_channels, kernel_size=3, stride=2, padding=1)
        
        # Simplified backbone structure
        # In real implementation, this would integrate with YOLOv8s layers
        self.layer1 = self._make_conv_block(base_channels, base_channels * 2, stride=2)
        self.layer2 = self._make_conv_block(base_channels * 2, base_channels * 4, stride=2)
        self.layer3 = self._make_conv_block(base_channels * 4, base_channels * 8, stride=2)
        
        # LFC: Feature compression
        self.lfc1 = LFC(base_channels * 8, base_channels * 4)  # 512→256
        
        self.layer4 = self._make_conv_block(base_channels * 4, base_channels * 8, stride=2)
        
        # LFC: Feature compression
        self.lfc2 = LFC(base_channels * 8, base_channels * 4)  # 512→256
        
    def _make_conv_block(self, in_channels, out_channels, stride=1):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                     stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True)
        )
    
    def forward(self, x):
        # Extract multi-scale features
        x0 = self.layer0(x)      # P1/2
        x1 = self.layer1(x0)     # P2/4
        x2 = self.layer2(x1)     # P3/8
        x3 = self.layer3(x2)     # P4/16
        x3 = self.lfc1(x3)       # LFC compression
        x4 = self.layer4(x3)     # P5/32
        x4 = self.lfc2(x4)       # LFC compression
        
        # Return multi-scale features for neck
        return [x2, x3, x4]  # P3, P4, P5


class YOLORDNeck(nn.Module):
    """
    YOLO-RD Neck with LGECA attention modules
    """
    def __init__(self, channels_list=[256, 512, 512]):
        super(YOLORDNeck, self).__init__()
        
        # Simplified neck structure (PANet-like)
        # P5 -> P4
        self.up1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv1 = self._make_conv_block(channels_list[2] + channels_list[1], 
                                           channels_list[1])
        
        # P4 -> P3
        self.up2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv2 = self._make_conv_block(channels_list[1] + channels_list[0], 
                                           channels_list[0])
        self.lgeca1 = LGECA(channels_list[0], reduction=16, alpha=0.5)
        
        # P3 -> P4
        self.down1 = self._make_conv_block(channels_list[0], channels_list[0], stride=2)
        self.conv3 = self._make_conv_block(channels_list[0] + channels_list[1], 
                                           channels_list[1])
        self.lgeca2 = LGECA(channels_list[1], reduction=16, alpha=0.5)
        
        # P4 -> P5
        self.down2 = self._make_conv_block(channels_list[1], channels_list[1], stride=2)
        self.conv4 = self._make_conv_block(channels_list[1] + channels_list[2], 
                                           channels_list[2])
        self.lgeca3 = LGECA(channels_list[2], reduction=16, alpha=0.5)
    
    def _make_conv_block(self, in_channels, out_channels, stride=1):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                     stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True)
        )
    
    def forward(self, features):
        """
        Args:
            features: List of [P3, P4, P5]
        Returns:
            List of enhanced features [P3, P4, P5]
        """
        p3, p4, p5 = features
        
        # Top-down pathway
        p5_up = self.up1(p5)
        p4_fused = torch.cat([p5_up, p4], dim=1)
        p4_fused = self.conv1(p4_fused)
        
        p4_up = self.up2(p4_fused)
        p3_fused = torch.cat([p4_up, p3], dim=1)
        p3_out = self.conv2(p3_fused)
        p3_out = self.lgeca1(p3_out)  # LGECA at P3
        
        # Bottom-up pathway
        p3_down = self.down1(p3_out)
        p4_fused2 = torch.cat([p3_down, p4_fused], dim=1)
        p4_out = self.conv3(p4_fused2)
        p4_out = self.lgeca2(p4_out)  # LGECA at P4
        
        p4_down = self.down2(p4_out)
        p5_fused = torch.cat([p4_down, p5], dim=1)
        p5_out = self.conv4(p5_fused)
        p5_out = self.lgeca3(p5_out)  # LGECA at P5
        
        return [p3_out, p4_out, p5_out]


class YOLORDHead(nn.Module):
    """
    YOLO-RD Detection Head
    """
    def __init__(self, num_classes=2, channels_list=[256, 512, 512]):
        super(YOLORDHead, self).__init__()
        self.num_classes = num_classes
        
        # Detection heads for each scale
        self.heads = nn.ModuleList([
            self._make_detection_head(ch, num_classes) for ch in channels_list
        ])
    
    def _make_detection_head(self, in_channels, num_classes):
        """Create detection head for one scale"""
        return nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(in_channels, (num_classes + 5), kernel_size=1)  # cls + box + obj
        )
    
    def forward(self, features):
        """
        Args:
            features: List of [P3, P4, P5]
        Returns:
            List of predictions for each scale
        """
        outputs = []
        for i, feat in enumerate(features):
            outputs.append(self.heads[i](feat))
        return outputs


class YOLORD(nn.Module):
    """
    Complete YOLO-RD Model
    Intégration de CSAF, LGECA, LFC et SR_WBCE_Loss
    Target: ~6.5M parameters, ~24.0 GFLOPs
    """
    def __init__(self, num_classes=2, in_channels=3, base_channels=64):
        super(YOLORD, self).__init__()
        
        self.num_classes = num_classes
        
        # Backbone with CSAF and LFC
        self.backbone = YOLORDBackbone(in_channels, base_channels)
        
        # Neck with LGECA
        self.neck = YOLORDNeck(channels_list=[
            base_channels * 4,   # 256 for P3
            base_channels * 4,   # 256 for P4 (after LFC)
            base_channels * 4    # 256 for P5 (after LFC)
        ])
        
        # Detection head
        self.head = YOLORDHead(num_classes, channels_list=[
            base_channels * 4,
            base_channels * 4,
            base_channels * 4
        ])
        
        # Loss function
        self.loss_fn = SR_WBCE_Loss(lambda1=0.5, lambda2=7.5, lambda3=1.5)
        
    def forward(self, x, targets=None):
        """
        Args:
            x: Input images [B, 3, H, W]
            targets: Ground truth (for training)
        Returns:
            If training: (loss, loss_dict)
            If inference: predictions
        """
        # Extract features
        backbone_features = self.backbone(x)
        
        # Enhance features with attention
        neck_features = self.neck(backbone_features)
        
        # Generate predictions
        predictions = self.head(neck_features)
        
        if self.training and targets is not None:
            # Format predictions for loss computation
            # predictions is a list of [P3, P4, P5] outputs
            # Each output has shape [B, num_classes+5, H, W]
            # For simplified loss computation, we'll flatten and split
            pred_dict = self._format_predictions(predictions)
            
            # Calculate loss during training
            return self.loss_fn(pred_dict, targets)
        else:
            # Return predictions during inference
            return predictions
    
    def _format_predictions(self, predictions):
        """
        Format model predictions for loss computation
        
        Args:
            predictions: List of [P3, P4, P5] predictions
        Returns:
            Dictionary with 'cls' and 'box' keys
        """
        # Concatenate all predictions
        all_preds = []
        for pred in predictions:
            B, C, H, W = pred.shape
            # Reshape to [B, H, W, C] then flatten
            pred_reshaped = pred.permute(0, 2, 3, 1).contiguous()
            pred_flat = pred_reshaped.view(-1, C)
            all_preds.append(pred_flat)
        
        all_preds = torch.cat(all_preds, dim=0)
        
        # Split into classification and box predictions
        # Format: [cls..., box(4), objectness]
        cls_pred = all_preds[:, :self.num_classes]
        box_pred = all_preds[:, self.num_classes:self.num_classes+4]
        
        return {
            'cls': cls_pred,
            'box': box_pred
        }
    
    def get_model_info(self):
        """Get model information (parameters, GFLOPs)"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'parameters_M': total_params / 1e6,
            'target_parameters_M': 6.5,
            'target_gflops': 24.0
        }


def create_yolo_rd_model(num_classes=2, pretrained=False):
    """
    Factory function to create YOLO-RD model
    
    Args:
        num_classes: Number of classes to detect
        pretrained: Whether to load pretrained weights
    
    Returns:
        YOLO-RD model instance
    """
    model = YOLORD(num_classes=num_classes)
    
    if pretrained:
        # TODO: Load pretrained weights
        warnings.warn("Pretrained weights not available yet")
    
    return model


# Example usage
if __name__ == "__main__":
    # Create model
    model = create_yolo_rd_model(num_classes=2)
    
    # Test forward pass
    x = torch.randn(1, 3, 640, 640)
    
    model.eval()
    with torch.no_grad():
        outputs = model(x)
    
    print("Model created successfully!")
    print(f"Model info: {model.get_model_info()}")
    print(f"Output scales: {len(outputs)}")
    for i, out in enumerate(outputs):
        print(f"  Scale {i}: {out.shape}")
