# YOLO-RD: Road Damage Detection Model

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Implémentation du modèle YOLO-RD basé sur YOLOv8s (Ultralytics, PyTorch) avec des modules personnalisés pour la détection des dommages routiers (fissures et nids-de-poule).

## 🎯 Objectifs

- **Réduction des paramètres**: ~6.5M paramètres (vs ~11M pour YOLOv8s)
- **Optimisation computationnelle**: ~24.0 GFLOPs (vs ~28.4 pour YOLOv8s)
- **Performance améliorée**: Mécanismes d'attention avancés pour la détection de petits objets

## 🏗️ Architecture

### Modules Personnalisés

#### 1. CSAF (Convolution Spatial-to-Depth Attention Fusion)
- **Position**: Couche 0 (remplace le premier bloc convolutionnel)
- **Fonction**: Fusion de deux branches de traitement avec attention ESE
  - Branche 1: Convolution 3x3 standard
  - Branche 2: SPD (Space-to-Depth) + convolutions
- **Avantage**: Préservation des informations fines dès les premières couches

#### 2. LGECA (Local-Global Enhanced Context Attention)
- **Position**: Couches 16, 20, 24 (entre neck et head)
- **Fonction**: Attention multi-échelle avec fusion adaptative
  - Branche globale: Capture le contexte global
  - Branche locale: Préserve les détails locaux
  - Fusion par paramètre α appris
- **Avantage**: Équilibre optimal entre contexte global et détails locaux

#### 3. LFC (Layer-wise Feature Compression)
- **Position**: Couches 7 et 10
- **Fonction**: Réduction des canaux (512→256)
- **Avantage**: Optimisation des paramètres sans perte significative de performance

#### 4. SR_WBCE_Loss (Scale-Robust Weighted BCE Loss)
- **Fonction**: Perte personnalisée pour la classification
- **Formule**: `L_total = λ₁·L_SR-BCE + λ₂·L_CIoU + λ₃·L_DFL`
- **Poids par défaut**: λ₁=0.5, λ₂=7.5, λ₃=1.5
- **Avantage**: Meilleure gestion des objets de différentes échelles

## 📦 Installation

```bash
# Clone le repository
git clone https://github.com/darouch-ikram/yolo-rd-colab.git
cd yolo-rd-colab

# Installation des dépendances
pip install -r requirements.txt
```

## 🚀 Utilisation

### 1. Google Colab (Recommandé)

Ouvrez le notebook `YOLO_RD_Colab.ipynb` dans Google Colab:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/darouch-ikram/yolo-rd-colab/blob/main/YOLO_RD_Colab.ipynb)

Le notebook inclut:
- Installation automatique des dépendances
- Chargement du dataset depuis Roboflow
- Création et test du modèle
- Configuration de l'entraînement

### 2. Utilisation Locale

#### Création du modèle

```python
from yolo_rd import create_yolo_rd_model

# Créer le modèle YOLO-RD
model = create_yolo_rd_model(num_classes=2)

# Afficher les informations
info = model.get_model_info()
print(f"Parameters: {info['parameters_M']:.2f}M")
```

#### Test des modules individuels

```python
from yolo_rd.modules import CSAF, LGECA, SR_WBCE_Loss
import torch

# Test CSAF
csaf = CSAF(in_channels=3, out_channels=64, kernel_size=3, stride=2)
x = torch.randn(1, 3, 640, 640)
output = csaf(x)
print(f"CSAF output shape: {output.shape}")

# Test LGECA
lgeca = LGECA(channels=256, reduction=16, alpha=0.5)
x = torch.randn(1, 256, 80, 80)
output = lgeca(x)
print(f"LGECA output shape: {output.shape}")

# Test Loss
loss_fn = SR_WBCE_Loss(lambda1=0.5, lambda2=7.5, lambda3=1.5)
pred = {'cls': torch.randn(10, 2), 'box': torch.randn(10, 4)}
target = {'cls': torch.randint(0, 2, (10, 2)).float(), 'box': torch.randn(10, 4)}
loss, loss_dict = loss_fn(pred, target)
print(f"Total loss: {loss.item():.4f}")
```

#### Entraînement avec Roboflow

```python
from yolo_rd.train import RoboflowDatasetLoader, YOLORDTrainer

# Télécharger le dataset
loader = RoboflowDatasetLoader(
    api_key="YOUR_ROBOFLOW_API_KEY",
    workspace="road-damage-detection-n2xkq",
    project="crack-and-pothole-bftyl"
)
dataset_path = loader.download_dataset()

# Créer le trainer
model = create_yolo_rd_model(num_classes=2)
trainer = YOLORDTrainer(model=model, config=config, device='cuda')

# Entraîner (nécessite data loaders)
# trainer.train(train_loader, val_loader, epochs=100)
```

## 📊 Dataset

Le projet utilise le dataset **Road Damage Detection** de Roboflow:
- **Source**: [Roboflow Universe](https://universe.roboflow.com/road-damage-detection-n2xkq/crack-and-pothole-bftyl)
- **Classes**: 2 (Crack, Pothole)
- **Format**: YOLOv8
- **Accès**: Via API Roboflow (pas de téléchargement local nécessaire)

## 🔧 Configuration

Les configurations sont disponibles dans `yolo_rd/models/config.py`:

```python
yolo_rd_simple_config = {
    'model_name': 'YOLO-RD',
    'num_classes': 2,
    'input_size': [640, 640],
    'custom_modules': {
        'CSAF': {'layer': 0, ...},
        'LGECA': {'layers': [18, 22, 26], ...},
        'LFC': {'layers': [7, 10], ...}
    },
    'loss': {
        'type': 'SR_WBCE_Loss',
        'lambda1': 0.5,
        'lambda2': 7.5,
        'lambda3': 1.5
    },
    'train': {
        'epochs': 100,
        'batch_size': 16,
        'lr0': 0.001,
        ...
    }
}
```

## 📁 Structure du Projet

```
yolo-rd-colab/
├── yolo_rd/
│   ├── __init__.py
│   ├── modules/
│   │   ├── __init__.py
│   │   ├── csaf.py          # Module CSAF
│   │   ├── lgeca.py         # Module LGECA
│   │   └── loss.py          # SR_WBCE_Loss
│   ├── models/
│   │   ├── __init__.py
│   │   ├── config.py        # Configurations
│   │   └── yolo_rd.py       # Modèle principal
│   └── train.py             # Script d'entraînement
├── YOLO_RD_Colab.ipynb      # Notebook Colab
├── requirements.txt         # Dépendances
└── README.md               # Documentation
```

## 🔬 Modules Techniques

### CSAF (csaf.py)
- `SPD`: Space-to-Depth transformation
- `ESE`: Effective Squeeze-and-Excitation attention
- `CSAF`: Module complet avec fusion

### LGECA (lgeca.py)
- `LGECA`: Attention local-global avec α adaptatif
- `LGECAv2`: Variante avec multi-échelle

### Loss (loss.py)
- `SR_BCE_Loss`: BCE robuste à l'échelle
- `DFL_Loss`: Distribution Focal Loss
- `SR_WBCE_Loss`: Perte complète combinée

### Model (yolo_rd.py)
- `LFC`: Compression de caractéristiques
- `YOLORDBackbone`: Backbone avec CSAF et LFC
- `YOLORDNeck`: Neck avec LGECA
- `YOLORDHead`: Tête de détection
- `YOLORD`: Modèle complet

## 📈 Performances Attendues

| Métrique | YOLO-RD | YOLOv8s |
|----------|---------|---------|
| Paramètres | ~6.5M | ~11M |
| GFLOPs | ~24.0 | ~28.4 |
| mAP@0.5 | TBD | Baseline |
| Vitesse | TBD | Baseline |

## 🤝 Contribution

Les contributions sont les bienvenues! N'hésitez pas à:
1. Fork le projet
2. Créer une branche (`git checkout -b feature/amelioration`)
3. Commit vos changements (`git commit -m 'Add amelioration'`)
4. Push vers la branche (`git push origin feature/amelioration`)
5. Ouvrir une Pull Request

## 📝 Citation

Si vous utilisez ce code dans vos recherches, veuillez citer:

```bibtex
@misc{yolord2024,
  title={YOLO-RD: Road Damage Detection with Enhanced Attention Mechanisms},
  author={Darouch, Ikram},
  year={2024},
  howpublished={\url{https://github.com/darouch-ikram/yolo-rd-colab}}
}
```

## 📄 Licence

Ce projet est sous licence MIT. Voir le fichier [LICENSE](LICENSE) pour plus de détails.

## 🙏 Remerciements

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) pour le framework de base
- [Roboflow](https://roboflow.com/) pour l'hébergement du dataset
- La communauté PyTorch pour les outils et ressources

## 📧 Contact

Pour toute question ou suggestion:
- GitHub Issues: [yolo-rd-colab/issues](https://github.com/darouch-ikram/yolo-rd-colab/issues)
- Email: [votre-email@example.com]

---

**Note**: Ce projet est en développement actif. Les performances et fonctionnalités peuvent évoluer.
