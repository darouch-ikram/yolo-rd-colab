"""
YOLO-RD Usage Examples
Demonstrates how to use the YOLO-RD model and its components
"""

import torch
from yolo_rd import create_yolo_rd_model, CSAF, LGECA, SR_WBCE_Loss


def example_1_basic_model():
    """Example 1: Create and use basic YOLO-RD model"""
    print("\n" + "="*60)
    print("Example 1: Basic Model Creation and Inference")
    print("="*60)
    
    # Create model
    model = create_yolo_rd_model(num_classes=2)
    model.eval()
    
    # Get model info
    info = model.get_model_info()
    print(f"\nModel Information:")
    print(f"  Total Parameters: {info['parameters_M']:.2f}M")
    print(f"  Trainable Parameters: {info['trainable_parameters'] / 1e6:.2f}M")
    print(f"  Target: {info['target_parameters_M']}M params, {info['target_gflops']} GFLOPs")
    
    # Create dummy input
    batch_size = 2
    input_tensor = torch.randn(batch_size, 3, 640, 640)
    
    # Run inference
    with torch.no_grad():
        outputs = model(input_tensor)
    
    print(f"\nInput shape: {input_tensor.shape}")
    print(f"Number of output scales: {len(outputs)}")
    for i, out in enumerate(outputs):
        print(f"  Scale {i} (P{i+3}): {out.shape}")
    
    print("\n✓ Basic inference successful!")


def example_2_csaf_module():
    """Example 2: Using CSAF module independently"""
    print("\n" + "="*60)
    print("Example 2: CSAF Module")
    print("="*60)
    
    # Create CSAF module
    csaf = CSAF(
        in_channels=3,
        out_channels=64,
        kernel_size=3,
        stride=2,
        padding=1
    )
    
    # Input image
    x = torch.randn(1, 3, 640, 640)
    
    # Forward pass
    output = csaf(x)
    
    print(f"\nCSAF Module:")
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Parameters: {sum(p.numel() for p in csaf.parameters()) / 1e3:.2f}K")
    
    print("\n✓ CSAF module working correctly!")


def example_3_lgeca_module():
    """Example 3: Using LGECA module independently"""
    print("\n" + "="*60)
    print("Example 3: LGECA Module")
    print("="*60)
    
    # Create LGECA module with custom alpha
    lgeca = LGECA(
        channels=256,
        reduction=16,
        alpha=0.5  # Learnable parameter
    )
    
    # Set to eval mode to avoid batch norm issues with batch_size=1
    lgeca.eval()
    
    # Feature tensor
    x = torch.randn(1, 256, 80, 80)
    
    # Forward pass
    with torch.no_grad():
        output = lgeca(x)
    
    print(f"\nLGECA Module:")
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Alpha (global-local balance): {lgeca.alpha.item():.4f}")
    print(f"  Parameters: {sum(p.numel() for p in lgeca.parameters()) / 1e3:.2f}K")
    
    # LGECA has residual connection, so output includes original features
    print(f"  Has residual connection: Yes")
    
    print("\n✓ LGECA module working correctly!")


def example_4_loss_function():
    """Example 4: Using SR_WBCE_Loss"""
    print("\n" + "="*60)
    print("Example 4: SR_WBCE_Loss Function")
    print("="*60)
    
    # Create loss function with custom weights
    loss_fn = SR_WBCE_Loss(
        lambda1=0.5,   # Classification weight
        lambda2=7.5,   # Localization weight
        lambda3=1.5    # DFL weight
    )
    
    # Dummy predictions and targets
    batch_size = 16
    predictions = {
        'cls': torch.randn(batch_size, 2),      # Classification logits
        'box': torch.randn(batch_size, 4)       # Box predictions [x, y, w, h]
    }
    
    targets = {
        'cls': torch.randint(0, 2, (batch_size, 2)).float(),
        'box': torch.randn(batch_size, 4)
    }
    
    # Calculate loss
    total_loss, loss_dict = loss_fn(predictions, targets)
    
    print(f"\nSR_WBCE_Loss:")
    print(f"  Total Loss: {total_loss.item():.4f}")
    print(f"  Components:")
    print(f"    - Classification (λ1={loss_fn.lambda1}): {loss_dict['cls'].item():.4f}")
    print(f"    - Localization (λ2={loss_fn.lambda2}): {loss_dict['box'].item():.4f}")
    print(f"    - DFL (λ3={loss_fn.lambda3}): {loss_dict['dfl'].item():.4f}")
    
    print("\n✓ Loss function working correctly!")


def example_5_training_setup():
    """Example 5: Setting up model for training"""
    print("\n" + "="*60)
    print("Example 5: Training Setup")
    print("="*60)
    
    # Create model
    model = create_yolo_rd_model(num_classes=2)
    
    # Set to training mode
    model.train()
    
    # Create optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=0.001,
        weight_decay=0.0005
    )
    
    # Create learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=100,
        eta_min=0.00001
    )
    
    # Dummy training batch
    images = torch.randn(4, 3, 640, 640)
    
    # Calculate required target size
    num_predictions = 4 * (80*80 + 40*40 + 20*20)  # batch * total anchors
    
    targets = {
        'cls': torch.randint(0, 2, (num_predictions, 2)).float(),
        'box': torch.randn(num_predictions, 4)
    }
    
    # Forward pass
    loss, loss_dict = model(images, targets)
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    print(f"\nTraining Setup:")
    print(f"  Batch size: {images.shape[0]}")
    print(f"  Image size: {images.shape[2:]}")
    print(f"  Total loss: {loss.item():.4f}")
    print(f"  Optimizer: AdamW (lr=0.001)")
    print(f"  Scheduler: CosineAnnealingLR")
    
    print("\n✓ Training setup working correctly!")


def example_6_multi_gpu():
    """Example 6: Multi-GPU setup (if available)"""
    print("\n" + "="*60)
    print("Example 6: Multi-GPU Setup")
    print("="*60)
    
    # Check GPU availability
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        print(f"\nGPU Information:")
        print(f"  Available GPUs: {device_count}")
        
        for i in range(device_count):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        
        # Create model
        model = create_yolo_rd_model(num_classes=2)
        
        # Move to GPU
        if device_count > 1:
            model = torch.nn.DataParallel(model)
            print(f"\n✓ Model wrapped with DataParallel for {device_count} GPUs")
        
        model = model.cuda()
        print(f"✓ Model moved to GPU(s)")
        
        # Test inference on GPU
        x = torch.randn(2, 3, 640, 640).cuda()
        model.eval()
        with torch.no_grad():
            outputs = model(x)
        
        print(f"\n✓ GPU inference successful!")
        
    else:
        print("\nNo GPU available. Running on CPU.")
        print("To use GPU, install CUDA-enabled PyTorch:")
        print("  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118")


def main():
    """Run all examples"""
    print("\n" + "="*70)
    print(" "*20 + "YOLO-RD Usage Examples")
    print("="*70)
    
    try:
        example_1_basic_model()
        example_2_csaf_module()
        example_3_lgeca_module()
        example_4_loss_function()
        example_5_training_setup()
        example_6_multi_gpu()
        
        print("\n" + "="*70)
        print("All examples completed successfully! 🎉")
        print("="*70)
        print("\nNext steps:")
        print("  1. Check out YOLO_RD_Colab.ipynb for interactive examples")
        print("  2. Read QUICKSTART.md for quick start guide")
        print("  3. See README.md for full documentation")
        print("  4. Run test_yolo_rd.py to verify installation")
        print("\nHappy detecting! 🛣️")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
