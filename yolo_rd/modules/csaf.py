"""
CSAF (Convolution Spatial-to-Depth Attention Fusion) Module
Remplace le premier bloc convolutionnel (couche 0) du backbone YOLOv8s
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SPD(nn.Module):
    """Space-to-Depth module"""
    def __init__(self, scale=2):
        super(SPD, self).__init__()
        self.scale = scale

    def forward(self, x):
        """
        Args:
            x: Input tensor [B, C, H, W]
        Returns:
            Output tensor [B, C * scale^2, H/scale, W/scale]
        """
        B, C, H, W = x.shape
        scale = self.scale
        
        # Ensure dimensions are divisible by scale
        assert H % scale == 0 and W % scale == 0, \
            f"Height ({H}) and Width ({W}) must be divisible by scale ({scale})"
        
        # Reshape: [B, C, H, W] -> [B, C, H/scale, scale, W/scale, scale]
        x = x.view(B, C, H // scale, scale, W // scale, scale)
        
        # Permute: [B, C, H/scale, scale, W/scale, scale] -> [B, C, scale, scale, H/scale, W/scale]
        x = x.permute(0, 1, 3, 5, 2, 4).contiguous()
        
        # Reshape to final shape: [B, C * scale^2, H/scale, W/scale]
        x = x.view(B, C * scale * scale, H // scale, W // scale)
        
        return x


class ESE(nn.Module):
    """Effective Squeeze-and-Excitation Attention Mechanism"""
    def __init__(self, channels, reduction=4):
        super(ESE, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        Args:
            x: Input tensor [B, C, H, W]
        Returns:
            Attention-weighted tensor [B, C, H, W]
        """
        B, C, _, _ = x.size()
        # Global average pooling: [B, C, H, W] -> [B, C, 1, 1]
        y = self.avg_pool(x).view(B, C)
        # Channel attention: [B, C] -> [B, C]
        y = self.fc(y).view(B, C, 1, 1)
        # Apply attention
        return x * y.expand_as(x)


class CSAF(nn.Module):
    """
    Convolution Spatial-to-Depth Attention Fusion Module
    Two-branch structure:
    - Branch 1: 3x3 Convolution
    - Branch 2: SPD + Convolutions
    Fusion via ESE attention mechanism
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=2, padding=1):
        super(CSAF, self).__init__()
        
        # Branch 1: Standard convolution path
        self.conv_branch = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, 
                     stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True)
        )
        
        # Branch 2: SPD path
        self.spd = SPD(scale=stride)
        # After SPD, channels are multiplied by stride^2
        spd_channels = in_channels * (stride ** 2)
        
        self.spd_conv = nn.Sequential(
            nn.Conv2d(spd_channels, out_channels, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, 
                     padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True)
        )
        
        # ESE attention for fusion
        self.ese_attention = ESE(out_channels * 2, reduction=4)
        
        # Final fusion convolution
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(out_channels * 2, out_channels, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True)
        )

    def forward(self, x):
        """
        Args:
            x: Input tensor [B, C, H, W]
        Returns:
            Fused output tensor [B, out_channels, H/stride, W/stride]
        """
        # Branch 1: Standard convolution
        conv_out = self.conv_branch(x)
        
        # Branch 2: SPD path
        spd_out = self.spd(x)
        spd_out = self.spd_conv(spd_out)
        
        # Concatenate branches
        concat = torch.cat([conv_out, spd_out], dim=1)
        
        # Apply ESE attention
        attended = self.ese_attention(concat)
        
        # Final fusion
        output = self.fusion_conv(attended)
        
        return output
