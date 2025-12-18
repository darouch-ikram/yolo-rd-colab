"""
YOLO-RD Model Configuration
Architecture basée sur YOLOv8s avec modules personnalisés
"""

# YOLO-RD Model Architecture
# Remplacements:
# - Layer 0: CSAF remplace le premier bloc convolutionnel
# - Layers 16, 20, 24: LGECA inséré entre neck et head
# - Layers 7, 8, 9: LFC (Layer-wise Feature Compression) 512→256 channels

yolo_rd_config = {
    # Model metadata
    'nc': 2,  # number of classes (crack and pothole)
    'depth_multiple': 0.33,  # model depth (YOLOv8s)
    'width_multiple': 0.50,  # model width (YOLOv8s)
    
    # Anchors (not used in YOLOv8, but kept for compatibility)
    'anchors': None,
    
    # Target parameters (~6.5M params, ~24.0 GFLOPs)
    'target_params': 6.5e6,
    'target_gflops': 24.0,
    
    # Backbone configuration
    'backbone': [
        # [from, repeats, module, args]
        [-1, 1, 'CSAF', [64, 3, 2, 1]],  # 0-P1/2 - CUSTOM: CSAF replaces Conv
        [-1, 1, 'Conv', [128, 3, 2]],    # 1-P2/4
        [-1, 3, 'C2f', [128, True]],     # 2
        [-1, 1, 'Conv', [256, 3, 2]],    # 3-P3/8
        [-1, 6, 'C2f', [256, True]],     # 4
        [-1, 1, 'Conv', [512, 3, 2]],    # 5-P4/16
        [-1, 6, 'C2f', [512, True]],     # 6
        [-1, 1, 'LFC', [256]],           # 7 - CUSTOM: Layer-wise Feature Compression
        [-1, 1, 'Conv', [512, 3, 2]],    # 8-P5/32 (before LFC, after: 256→512)
        [-1, 3, 'C2f', [512, True]],     # 9
        [-1, 1, 'LFC', [256]],           # 10 - CUSTOM: Layer-wise Feature Compression
        [-1, 1, 'SPPF', [512, 5]],       # 11
    ],
    
    # Head configuration
    'head': [
        [-1, 1, 'Upsample', [None, 2, 'nearest']],  # 12
        [[-1, 6], 1, 'Concat', [1]],                # 13 - cat backbone P4
        [-1, 3, 'C2f', [512, False]],               # 14
        
        [-1, 1, 'Upsample', [None, 2, 'nearest']],  # 15
        [[-1, 4], 1, 'Concat', [1]],                # 16 - cat backbone P3
        [-1, 3, 'C2f', [256, False]],               # 17
        [-1, 1, 'LGECA', [256]],                    # 18 - CUSTOM: LGECA attention
        
        [-1, 1, 'Conv', [256, 3, 2]],               # 19
        [[-1, 14], 1, 'Concat', [1]],               # 20 - cat head P4
        [-1, 3, 'C2f', [512, False]],               # 21
        [-1, 1, 'LGECA', [512]],                    # 22 - CUSTOM: LGECA attention
        
        [-1, 1, 'Conv', [512, 3, 2]],               # 23
        [[-1, 11], 1, 'Concat', [1]],               # 24 - cat head P5
        [-1, 3, 'C2f', [512, False]],               # 25
        [-1, 1, 'LGECA', [512]],                    # 26 - CUSTOM: LGECA attention
        
        [[18, 22, 26], 1, 'Detect', ['nc']],        # 27 - Detect(P3, P4, P5)
    ]
}

# Simplified configuration for easier parsing
yolo_rd_simple_config = {
    'model_name': 'YOLO-RD',
    'base_model': 'YOLOv8s',
    'input_size': [640, 640],
    'num_classes': 2,
    
    # Custom modules positions
    'custom_modules': {
        'CSAF': {
            'layer': 0,
            'replaces': 'Conv',
            'args': {'in_channels': 3, 'out_channels': 64, 'kernel_size': 3, 'stride': 2}
        },
        'LGECA': {
            'layers': [18, 22, 26],  # After neck layers
            'channels': [256, 512, 512]
        },
        'LFC': {
            'layers': [7, 10],  # Feature compression
            'target_channels': 256
        }
    },
    
    # Loss configuration
    'loss': {
        'type': 'SR_WBCE_Loss',
        'lambda1': 0.5,   # SR-BCE weight
        'lambda2': 7.5,   # CIoU weight
        'lambda3': 1.5    # DFL weight
    },
    
    # Training configuration
    'train': {
        'epochs': 100,
        'batch_size': 16,
        'img_size': 640,
        'optimizer': 'AdamW',
        'lr0': 0.001,
        'lrf': 0.01,
        'momentum': 0.937,
        'weight_decay': 0.0005,
        'warmup_epochs': 3,
        'warmup_momentum': 0.8,
        'warmup_bias_lr': 0.1,
    },
    
    # Data augmentation
    'augmentation': {
        'hsv_h': 0.015,
        'hsv_s': 0.7,
        'hsv_v': 0.4,
        'degrees': 0.0,
        'translate': 0.1,
        'scale': 0.5,
        'shear': 0.0,
        'perspective': 0.0,
        'flipud': 0.0,
        'fliplr': 0.5,
        'mosaic': 1.0,
        'mixup': 0.0,
    }
}
