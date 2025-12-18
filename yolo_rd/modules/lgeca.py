"""
LGECA (Local-Global Enhanced Context Attention) Module
Inséré entre les couches neck et head (couches 16, 20 et 24)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LGECA(nn.Module):
    """
    Local-Global Enhanced Context Attention Module
    Two-branch structure:
    - Global branch: Captures global context with adaptive pooling
    - Local branch: Captures local details with depth-wise convolutions
    Fusion with adaptive weights based on alpha parameter
    """
    def __init__(self, channels, reduction=16, alpha=0.5):
        """
        Args:
            channels: Number of input/output channels
            reduction: Reduction ratio for channel attention
            alpha: Weight parameter for balancing global and local features (0 to 1)
        """
        super(LGECA, self).__init__()
        self.channels = channels
        self.alpha = nn.Parameter(torch.tensor(alpha), requires_grad=True)
        
        # Global branch
        self.global_branch = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels // reduction),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.Sigmoid()
        )
        
        # Local branch - Depth-wise separable convolutions
        self.local_branch = nn.Sequential(
            # Depth-wise convolution
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, 
                     groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            # Point-wise convolution
            nn.Conv2d(channels, channels // reduction, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels // reduction),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.Sigmoid()
        )
        
        # Optional: Additional convolution for feature refinement
        self.refine_conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True)
        )

    def forward(self, x):
        """
        Args:
            x: Input tensor [B, C, H, W]
        Returns:
            Enhanced feature tensor [B, C, H, W]
        """
        # Global context attention
        global_att = self.global_branch(x)
        
        # Local context attention
        local_att = self.local_branch(x)
        
        # Adaptive fusion using alpha parameter
        # alpha is learnable and clamped between 0 and 1
        alpha = torch.clamp(self.alpha, 0.0, 1.0)
        
        # Combined attention weights
        combined_att = alpha * global_att + (1 - alpha) * local_att
        
        # Apply attention to input features
        enhanced = x * combined_att
        
        # Optional refinement
        output = self.refine_conv(enhanced)
        
        # Residual connection
        output = output + x
        
        return output


class LGECAv2(nn.Module):
    """
    Alternative implementation of LGECA with more explicit multi-scale processing
    """
    def __init__(self, channels, reduction=16, alpha=0.5):
        super(LGECAv2, self).__init__()
        self.channels = channels
        self.alpha = nn.Parameter(torch.tensor(alpha), requires_grad=True)
        
        # Global branch with multiple pooling scales
        self.global_pools = nn.ModuleList([
            nn.AdaptiveAvgPool2d(1),
            nn.AdaptiveAvgPool2d(3),
            nn.AdaptiveAvgPool2d(5)
        ])
        
        self.global_fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels // reduction),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, kernel_size=1, bias=False),
            nn.Sigmoid()
        )
        
        # Local branch with multi-kernel convolutions
        self.local_conv3 = nn.Conv2d(channels, channels // 3, kernel_size=3, 
                                     padding=1, groups=channels // 3, bias=False)
        self.local_conv5 = nn.Conv2d(channels, channels // 3, kernel_size=5, 
                                     padding=2, groups=channels // 3, bias=False)
        self.local_conv7 = nn.Conv2d(channels, channels // 3, kernel_size=7, 
                                     padding=3, groups=channels // 3, bias=False)
        
        self.local_fusion = nn.Sequential(
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        B, C, H, W = x.shape
        
        # Global branch - multi-scale pooling
        global_feats = []
        for pool in self.global_pools:
            pooled = pool(x)
            global_feats.append(F.interpolate(pooled, size=(H, W), mode='bilinear', 
                                             align_corners=False))
        global_feat = sum(global_feats) / len(global_feats)
        global_att = self.global_fc(global_feat)
        
        # Local branch - multi-kernel processing
        local_feat = torch.cat([
            self.local_conv3(x[:, :C//3, :, :]),
            self.local_conv5(x[:, C//3:2*C//3, :, :]),
            self.local_conv7(x[:, 2*C//3:, :, :])
        ], dim=1)
        local_att = self.local_fusion(local_feat)
        
        # Adaptive fusion
        alpha = torch.clamp(self.alpha, 0.0, 1.0)
        combined_att = alpha * global_att + (1 - alpha) * local_att
        
        # Apply attention
        output = x * combined_att + x
        
        return output
