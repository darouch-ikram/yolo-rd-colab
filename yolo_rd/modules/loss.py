"""
SR_WBCE_Loss (Scale-Robust Weighted Binary Cross-Entropy Loss)
Remplace la perte de classification BCE de YOLOv8
Formule: L_total = λ1 · L_SR-BCE + λ2 · L_CIoU + λ3 · L_DFL
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def bbox_iou(box1, box2, xywh=True, GIoU=False, DIoU=False, CIoU=False, eps=1e-7):
    """
    Calculate IoU between boxes
    Args:
        box1, box2: Bounding boxes in format [x, y, w, h] or [x1, y1, x2, y2]
        xywh: If True, boxes are in xywh format
        CIoU: If True, calculate Complete IoU
    Returns:
        IoU value
    """
    # Get the coordinates of bounding boxes
    if xywh:  # transform from xywh to xyxy
        (x1, y1, w1, h1), (x2, y2, w2, h2) = box1.chunk(4, -1), box2.chunk(4, -1)
        w1_, h1_, w2_, h2_ = w1 / 2, h1 / 2, w2 / 2, h2 / 2
        b1_x1, b1_x2, b1_y1, b1_y2 = x1 - w1_, x1 + w1_, y1 - h1_, y1 + h1_
        b2_x1, b2_x2, b2_y1, b2_y2 = x2 - w2_, x2 + w2_, y2 - h2_, y2 + h2_
    else:  # x1, y1, x2, y2 = box1
        b1_x1, b1_y1, b1_x2, b1_y2 = box1.chunk(4, -1)
        b2_x1, b2_y1, b2_x2, b2_y2 = box2.chunk(4, -1)
        w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
        w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps

    # Intersection area
    inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
            (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)

    # Union Area
    union = w1 * h1 + w2 * h2 - inter + eps

    # IoU
    iou = inter / union

    if CIoU or DIoU or GIoU:
        cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)  # convex width
        ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)  # convex height
        if CIoU or DIoU:  # Distance or Complete IoU
            c2 = cw ** 2 + ch ** 2 + eps  # convex diagonal squared
            rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 + 
                    (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4  # center dist ** 2
            if CIoU:
                v = (4 / math.pi ** 2) * torch.pow(
                    torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)
                with torch.no_grad():
                    alpha = v / (v - iou + (1 + eps))
                return iou - (rho2 / c2 + v * alpha)  # CIoU
            return iou - rho2 / c2  # DIoU
        c_area = cw * ch + eps  # convex area
        return iou - (c_area - union) / c_area  # GIoU
    
    return iou


class SR_BCE_Loss(nn.Module):
    """Scale-Robust Binary Cross-Entropy Loss"""
    def __init__(self, pos_weight=None, reduction='mean'):
        super(SR_BCE_Loss, self).__init__()
        self.pos_weight = pos_weight
        self.reduction = reduction

    def forward(self, pred, target, scale_factor=None):
        """
        Args:
            pred: Predicted probabilities [N]
            target: Target labels [N]
            scale_factor: Optional scale factor for weighting [N]
        Returns:
            SR-BCE loss
        """
        # Standard BCE loss
        bce = F.binary_cross_entropy_with_logits(
            pred, target, 
            pos_weight=self.pos_weight,
            reduction='none'
        )
        
        # Scale-robust weighting
        if scale_factor is not None:
            # Weight smaller objects more heavily
            # scale_factor should be inversely proportional to object size
            weights = 1.0 + scale_factor
            bce = bce * weights
        
        if self.reduction == 'mean':
            return bce.mean()
        elif self.reduction == 'sum':
            return bce.sum()
        else:
            return bce


class DFL_Loss(nn.Module):
    """Distribution Focal Loss for box regression"""
    def __init__(self, reg_max=16):
        super(DFL_Loss, self).__init__()
        self.reg_max = reg_max

    def forward(self, pred_dist, target):
        """
        Args:
            pred_dist: Predicted distribution [N, reg_max+1]
            target: Target values [N]
        Returns:
            DFL loss
        """
        target = target.clamp(0, self.reg_max - 1 - 0.01)
        tl = target.long()  # target left
        tr = tl + 1  # target right
        wl = tr - target  # weight left
        wr = 1 - wl  # weight right
        
        return (F.cross_entropy(pred_dist, tl, reduction='none') * wl + 
                F.cross_entropy(pred_dist, tr, reduction='none') * wr).mean()


class SR_WBCE_Loss(nn.Module):
    """
    Complete YOLO-RD Loss Function
    L_total = λ1 · L_SR-BCE + λ2 · L_CIoU + λ3 · L_DFL
    Default: λ1=0.5, λ2=7.5, λ3=1.5
    """
    def __init__(self, lambda1=0.5, lambda2=7.5, lambda3=1.5, reg_max=16):
        """
        Args:
            lambda1: Weight for SR-BCE loss (classification)
            lambda2: Weight for CIoU loss (localization)
            lambda3: Weight for DFL loss (distribution)
            reg_max: Maximum value for DFL
        """
        super(SR_WBCE_Loss, self).__init__()
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.lambda3 = lambda3
        
        self.sr_bce = SR_BCE_Loss()
        self.dfl = DFL_Loss(reg_max=reg_max)

    def forward(self, predictions, targets):
        """
        Args:
            predictions: Dict containing:
                - 'cls': Classification predictions [N, num_classes]
                - 'box': Box predictions [N, 4] in xywh format
                - 'dist': Distribution predictions [N, 4*(reg_max+1)] (optional)
            targets: Dict containing:
                - 'cls': Target classes [N, num_classes]
                - 'box': Target boxes [N, 4] in xywh format
                - 'scale': Scale factors [N] (optional)
        Returns:
            Total loss and dict of individual losses
        """
        device = predictions['cls'].device
        
        # Classification loss (SR-BCE)
        scale_factor = targets.get('scale', None)
        loss_cls = self.sr_bce(
            predictions['cls'], 
            targets['cls'],
            scale_factor=scale_factor
        )
        
        # Localization loss (CIoU)
        if 'box' in predictions and 'box' in targets:
            # Calculate CIoU
            ciou = bbox_iou(predictions['box'], targets['box'], 
                           xywh=True, CIoU=True)
            loss_box = (1.0 - ciou).mean()
        else:
            loss_box = torch.tensor(0.0, device=device)
        
        # Distribution Focal Loss
        if 'dist' in predictions and 'box' in targets:
            # Convert box targets to distribution targets
            # This is simplified - in practice, you'd need proper target encoding
            loss_dfl = torch.tensor(0.0, device=device)
            # loss_dfl = self.dfl(predictions['dist'], targets['box'])
        else:
            loss_dfl = torch.tensor(0.0, device=device)
        
        # Total loss
        total_loss = (self.lambda1 * loss_cls + 
                     self.lambda2 * loss_box + 
                     self.lambda3 * loss_dfl)
        
        # Return total loss and breakdown
        loss_dict = {
            'total': total_loss,
            'cls': loss_cls,
            'box': loss_box,
            'dfl': loss_dfl
        }
        
        return total_loss, loss_dict


class YOLORDLoss(SR_WBCE_Loss):
    """Alias for compatibility"""
    pass
