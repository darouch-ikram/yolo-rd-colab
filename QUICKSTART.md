# YOLO-RD Quick Start Guide

## 🚀 Get Started in 5 Minutes

### Option 1: Google Colab (Recommended for Beginners)

1. **Open the Colab Notebook**
   
   Click here: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/darouch-ikram/yolo-rd-colab/blob/main/YOLO_RD_Colab.ipynb)

2. **Run All Cells**
   
   Just click `Runtime` → `Run all` and watch the magic happen!

3. **Get Your Roboflow API Key**
   
   - Go to https://app.roboflow.com/
   - Sign up/login (free)
   - Navigate to Settings → API Keys
   - Copy your key and paste it in the notebook

That's it! The notebook will:
- ✅ Install all dependencies
- ✅ Download the dataset
- ✅ Create the YOLO-RD model
- ✅ Run all tests
- ✅ Show you model statistics

---

### Option 2: Local Installation (For Developers)

```bash
# 1. Clone the repository
git clone https://github.com/darouch-ikram/yolo-rd-colab.git
cd yolo-rd-colab

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run tests to verify installation
python test_yolo_rd.py

# 4. Create a simple test script
cat > test.py << 'EOF'
from yolo_rd import create_yolo_rd_model
import torch

# Create model
model = create_yolo_rd_model(num_classes=2)
print(f"Model created: {model.get_model_info()}")

# Test inference
x = torch.randn(1, 3, 640, 640)
model.eval()
with torch.no_grad():
    outputs = model(x)
print(f"✓ Model works! Output shapes: {[o.shape for o in outputs]}")
EOF

# 5. Run it
python test.py
```

---

## 📦 What's Included?

### Core Modules

```python
from yolo_rd.modules import CSAF, LGECA, SR_WBCE_Loss

# CSAF: Spatial-to-Depth Attention Fusion
csaf = CSAF(in_channels=3, out_channels=64)

# LGECA: Local-Global Enhanced Context Attention
lgeca = LGECA(channels=256, alpha=0.5)

# SR_WBCE_Loss: Scale-Robust Weighted BCE Loss
loss_fn = SR_WBCE_Loss(lambda1=0.5, lambda2=7.5, lambda3=1.5)
```

### Complete Model

```python
from yolo_rd import create_yolo_rd_model

# Create YOLO-RD model
model = create_yolo_rd_model(num_classes=2)

# Get model info
info = model.get_model_info()
print(f"Parameters: {info['parameters_M']:.2f}M")
# Output: Parameters: 10.99M (target: 6.5M)

# Use for inference
import torch
model.eval()
image = torch.randn(1, 3, 640, 640)
predictions = model(image)
```

---

## 🎯 Key Features

| Feature | Status | Description |
|---------|--------|-------------|
| **CSAF Module** | ✅ | Replaces first conv layer, preserves fine details |
| **LGECA Module** | ✅ | Attention at layers 16, 20, 24 for enhanced context |
| **LFC** | ✅ | Layer-wise compression (512→256 channels) |
| **SR_WBCE_Loss** | ✅ | Custom loss with λ₁=0.5, λ₂=7.5, λ₃=1.5 |
| **Roboflow Integration** | ✅ | Direct dataset loading from cloud |
| **Google Colab** | ✅ | Ready-to-use notebook with all examples |

---

## 📊 Model Architecture

```
Input (3, 640, 640)
    ↓
[Layer 0] CSAF → (64, 320, 320)
    ↓
Backbone (YOLOv8s-inspired)
    ↓
[Layers 7, 10] LFC (512→256)
    ↓
Neck (PANet-style)
    ↓
[Layers 18, 22, 26] LGECA Attention
    ↓
Head → Multi-scale Predictions
    ↓
P3: (2, 80, 80)  # Fine details
P4: (2, 40, 40)  # Medium objects
P5: (2, 20, 20)  # Large objects
```

---

## 🧪 Run Tests

```bash
# Run all tests
python test_yolo_rd.py

# Expected output:
# ✓ CSAF Module ......... PASSED
# ✓ LGECA Module ........ PASSED  
# ✓ SR_WBCE_Loss ........ PASSED
# ✓ YOLO-RD Model ....... PASSED
# ✓ Training Mode ....... PASSED
# Total: 5/5 tests passed
```

---

## 📖 Next Steps

1. **For Research**: Read the detailed README.md
2. **For Training**: Check out `yolo_rd/train.py`
3. **For Customization**: Explore `yolo_rd/models/config.py`
4. **For Integration**: See examples in `YOLO_RD_Colab.ipynb`

---

## 🆘 Troubleshooting

### Problem: "torch not found"
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118  # For CUDA 11.8
# or
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu   # For CPU only
```

### Problem: "Roboflow API key invalid"
1. Go to https://app.roboflow.com/
2. Settings → API Keys
3. Generate new key if needed
4. Make sure you're using the correct workspace/project names

### Problem: Model parameters > 6.5M
This is expected in the current implementation. The model has ~11M parameters.
To reduce to 6.5M target:
- Reduce base_channels from 64 to 48
- Apply more aggressive LFC
- Use depth_multiple=0.25 instead of 0.33

---

## 💡 Pro Tips

1. **GPU Memory**: If you run out of GPU memory, reduce batch_size in config
2. **Speed**: Use mixed precision training: `torch.cuda.amp.autocast()`
3. **Dataset**: You can use your own dataset - just format it like YOLO
4. **Fine-tuning**: Load YOLOv8s weights as starting point (coming soon)

---

## 📚 Learn More

- Full documentation: [README.md](README.md)
- Architecture details: `yolo_rd/models/config.py`
- Module implementations: `yolo_rd/modules/`
- Training examples: `yolo_rd/train.py`

---

## 🎉 You're Ready!

Start with the Colab notebook and have fun detecting road damage! 🛣️

Questions? Open an issue on GitHub!
