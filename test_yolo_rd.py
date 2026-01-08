"""
Test script to validate YOLO-RD implementation
"""

import torch
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from yolo_rd.modules import CSAF, LGECA, SR_WBCE_Loss
from yolo_rd.models import YOLORD, create_yolo_rd_model


def test_csaf():
    """Test CSAF module"""
    print("\n" + "="*50)
    print("Testing CSAF Module")
    print("="*50)
    
    csaf = CSAF(in_channels=3, out_channels=64, kernel_size=3, stride=2)
    x = torch.randn(2, 3, 640, 640)
    
    try:
        output = csaf(x)
        expected_shape = (2, 64, 320, 320)
        assert output.shape == expected_shape, f"Expected {expected_shape}, got {output.shape}"
        print(f"✓ Input shape: {x.shape}")
        print(f"✓ Output shape: {output.shape}")
        print(f"✓ CSAF test passed!")
        return True
    except Exception as e:
        print(f"✗ CSAF test failed: {e}")
        return False


def test_lgeca():
    """Test LGECA module"""
    print("\n" + "="*50)
    print("Testing LGECA Module")
    print("="*50)
    
    lgeca = LGECA(channels=256, reduction=16, alpha=0.5)
    x = torch.randn(2, 256, 80, 80)
    
    try:
        output = lgeca(x)
        expected_shape = (2, 256, 80, 80)
        assert output.shape == expected_shape, f"Expected {expected_shape}, got {output.shape}"
        print(f"✓ Input shape: {x.shape}")
        print(f"✓ Output shape: {output.shape}")
        print(f"✓ Alpha parameter: {lgeca.alpha.item():.4f}")
        print(f"✓ LGECA test passed!")
        return True
    except Exception as e:
        print(f"✗ LGECA test failed: {e}")
        return False


def test_loss():
    """Test SR_WBCE_Loss"""
    print("\n" + "="*50)
    print("Testing SR_WBCE_Loss")
    print("="*50)
    
    loss_fn = SR_WBCE_Loss(lambda1=0.5, lambda2=7.5, lambda3=1.5)
    
    pred = {
        'cls': torch.randn(10, 2),
        'box': torch.randn(10, 4)
    }
    target = {
        'cls': torch.randint(0, 2, (10, 2)).float(),
        'box': torch.randn(10, 4)
    }
    
    try:
        loss, loss_dict = loss_fn(pred, target)
        print(f"✓ Total loss: {loss.item():.4f}")
        print(f"✓ Classification loss: {loss_dict['cls'].item():.4f}")
        print(f"✓ Localization loss: {loss_dict['box'].item():.4f}")
        print(f"✓ DFL loss: {loss_dict['dfl'].item():.4f}")
        print(f"✓ Loss function test passed!")
        return True
    except Exception as e:
        print(f"✗ Loss function test failed: {e}")
        return False


def test_model():
    """Test complete YOLO-RD model"""
    print("\n" + "="*50)
    print("Testing YOLO-RD Model")
    print("="*50)
    
    try:
        # Create model
        model = create_yolo_rd_model(num_classes=2)
        print("✓ Model created successfully")
        
        # Get model info
        info = model.get_model_info()
        print(f"✓ Total parameters: {info['parameters_M']:.2f}M")
        print(f"✓ Trainable parameters: {info['trainable_parameters'] / 1e6:.2f}M")
        print(f"✓ Target: {info['target_parameters_M']}M params, {info['target_gflops']} GFLOPs")
        
        # Test forward pass
        model.eval()
        x = torch.randn(2, 3, 640, 640)
        
        with torch.no_grad():
            outputs = model(x)
        
        print(f"✓ Forward pass successful")
        print(f"✓ Number of output scales: {len(outputs)}")
        for i, out in enumerate(outputs):
            print(f"  - Scale {i}: {out.shape}")
        
        print(f"✓ Model test passed!")
        return True
        
    except Exception as e:
        print(f"✗ Model test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_model_training_mode():
    """Test model in training mode with loss"""
    print("\n" + "="*50)
    print("Testing YOLO-RD Model (Training Mode)")
    print("="*50)
    
    try:
        model = create_yolo_rd_model(num_classes=2)
        model.train()
        
        x = torch.randn(2, 3, 640, 640)
        
        # Get prediction size to create matching targets
        model.eval()
        with torch.no_grad():
            test_outputs = model(x)
            # Calculate total prediction size
            total_preds = sum([out.numel() // (2 + 5) for out in test_outputs])
        
        model.train()
        
        # Create dummy targets with correct size
        # For a 640x640 image, we expect ~16800 predictions across all scales
        # P3: 80x80, P4: 40x40, P5: 20x20
        num_predictions = 2 * (80*80 + 40*40 + 20*20)  # batch_size * total_anchors
        
        targets = {
            'cls': torch.randint(0, 2, (num_predictions, 2)).float(),
            'box': torch.randn(num_predictions, 4)
        }
        
        # Forward pass with targets
        loss, loss_dict = model(x, targets)
        
        print(f"✓ Training mode forward pass successful")
        print(f"✓ Total loss: {loss.item():.4f}")
        print(f"✓ Loss components:")
        for k, v in loss_dict.items():
            if torch.is_tensor(v):
                print(f"  - {k}: {v.item():.4f}")
        
        # Test backward pass
        loss.backward()
        print(f"✓ Backward pass successful")
        
        print(f"✓ Training mode test passed!")
        return True
        
    except Exception as e:
        print(f"✗ Training mode test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("\n" + "="*70)
    print(" "*15 + "YOLO-RD Implementation Tests")
    print("="*70)
    
    results = {
        'CSAF Module': test_csaf(),
        'LGECA Module': test_lgeca(),
        'SR_WBCE_Loss': test_loss(),
        'YOLO-RD Model': test_model(),
        'Training Mode': test_model_training_mode()
    }
    
    # Summary
    print("\n" + "="*70)
    print("Test Summary")
    print("="*70)
    
    passed = sum(results.values())
    total = len(results)
    
    for name, result in results.items():
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"{name:.<50} {status}")
    
    print("="*70)
    print(f"Total: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! YOLO-RD implementation is working correctly.")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Please check the errors above.")
        return 1


if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)
