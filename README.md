# SRGAN — Super-Resolution sur dataset personnalisé

Implémentation PyTorch de [SRGAN](https://arxiv.org/abs/1609.04802) (Photo-Realistic Single Image Super-Resolution) entraînable sur n'importe quel dataset d'images HR/LR.

---

## Installation

### 1. Créer l'environnement conda

```bash
conda env create -f environment.yml
conda activate srganenv
```

### 2. Installer les dépendances Python

```bash
pip install -r requirements.txt
```

---

## Structure du projet

```
SRGAN_CustomDataset/
├── configs/
│   └── default.yaml       # Tous les hyperparamètres
├── scripts/
│   └── generate_lr.py     # Génération d'images LR depuis des HR
├── custom_dataset/        # Vos images d'entraînement (à placer ici)
├── pretrained_models/     # Modèles pré-entraînés sur DIV2K
├── checkpoints/           # Sauvegardes générées pendant l'entraînement
├── logs/                  # Logs TensorBoard
├── result/                # Images produites en test/inférence
├── config.py              # Chargement de la configuration YAML
├── dataset.py             # Classes Dataset PyTorch
├── losses.py              # PerceptualLoss, TVLoss
├── main.py                # Point d'entrée CLI
├── ops.py                 # Blocs de convolution réutilisables
├── srgan_model.py         # Architectures Generator et Discriminator
├── trainer.py             # Boucles d'entraînement et d'évaluation
└── vgg19.py               # VGG19 pour la loss perceptuelle
```

---

## Préparer ses données

Le dataset doit contenir deux dossiers : un avec les images haute résolution (HR) et un avec les images basse résolution (LR, facteur ×4 par défaut).

```
custom_dataset/
├── HR/          # Images originales (ex. 96×96 ou plus)
└── LR/          # Images dégradées (ex. 24×24, soit HR / 4)
```

Les fichiers doivent être triés de façon identique dans les deux dossiers (même ordre = même paire).

### Option A — Utiliser le dataset DIV2K complet

Le repo inclut les 800 images LR ×4 (`DIV2K_train_LR_bicubic-2/X4`). Les images HR correspondantes doivent être téléchargées séparément depuis le site officiel DIV2K et placées dans :

```
custom_dataset/DIV2K_train_HR/   ← 0001.png … 0800.png
```

Puis mettre à jour `configs/default.yaml` :

```yaml
LR_path: ./custom_dataset/DIV2K_train_LR_bicubic-2/X4
GT_path: ./custom_dataset/DIV2K_train_HR
```

### Option B — Générer les LR depuis ses propres images HR (test rapide)

Si vous n'avez que des images HR (comme `hr_valid_HR`), le script `generate_lr.py` crée les images LR correspondantes par sous-échantillonnage bicubique :

```bash
python scripts/generate_lr.py \
    --hr_dir custom_dataset/hr_valid_HR \
    --lr_dir custom_dataset/hr_valid_LR_x4 \
    --scale 4
```

Puis mettre à jour `configs/default.yaml` :

```yaml
LR_path: ./custom_dataset/hr_valid_LR_x4
GT_path: ./custom_dataset/hr_valid_HR
```

---

## Entraînement

L'entraînement se déroule en deux phases automatiquement enchaînées :

1. **Phase 1 — SRResNet** : pré-entraînement avec une loss L2 uniquement (génère un modèle stable)
2. **Phase 2 — SRGAN** : fine-tuning avec loss perceptuelle + adversariale (améliore le réalisme)

### Lancer depuis zéro

```bash
python main.py --mode train
```

### Utiliser un fichier de config différent

```bash
python main.py --config configs/default.yaml --mode train
```

### Surcharger des paramètres à la volée

```bash
python main.py --mode train --batch_size 8 --pre_train_epoch 2000
```

### Limiter le nombre d'epochs (test rapide)

Pour un entraînement court sans modifier le config, passez les epochs directement en argument :

```bash
# 200 epochs de pré-entraînement + 100 epochs de fine-tuning
python main.py --mode train --pre_train_epoch 200 --fine_train_epoch 100
```

Pour arrêter un entraînement en cours : **Ctrl+C**. Le dernier checkpoint sauvegardé permet de reprendre avec `--resume`.

### Reprendre un entraînement interrompu

```bash
# Reprendre depuis un checkpoint SRResNet (phase 1)
python main.py --resume checkpoints/srresnet_epoch0800.pt

# Reprendre depuis un checkpoint SRGAN (phase 2)
python main.py --resume checkpoints/srgan_epoch0500.pt
```

### Suivre l'entraînement avec TensorBoard

```bash
tensorboard --logdir logs/
```

Puis ouvrir `http://localhost:6006` dans un navigateur.

---

## Hyperparamètres principaux

Tous les paramètres sont dans `configs/default.yaml`. Les plus importants :

| Paramètre | Défaut | Description |
|---|---|---|
| `pre_train_epoch` | 8000 | Epochs phase SRResNet (L2) |
| `fine_train_epoch` | 4000 | Epochs phase SRGAN (perceptuel + adv) |
| `batch_size` | 16 | Taille du batch |
| `patch_size` | 24 | Taille du patch LR extrait pour l'entraînement |
| `scale` | 4 | Facteur d'upscaling (×4) |
| `lr` | 1e-4 | Learning rate initial |
| `L2_coeff` | 1.0 | Poids de la loss L2 |
| `adv_coeff` | 1e-3 | Poids de la loss adversariale |
| `vgg_rescale_coeff` | 0.006 | Poids de la loss perceptuelle |
| `feat_layer` | `relu5_4` | Couche VGG19 pour la loss perceptuelle |

---

## Choisir le bon checkpoint pour l'inférence

L'entraînement produit deux types de checkpoints dans `checkpoints/` :

| Fichier | Phase | Description |
|---|---|---|
| `srresnet_epoch<N>.pt` | Phase 1 (L2) | Images nettes, artefacts réduits, moins de détails fins |
| `srgan_epoch<N>.pt` | Phase 2 (SRGAN) | Textures plus réalistes, meilleure qualité visuelle |

**Pour l'inférence, utilisez toujours le checkpoint SRGAN** (`srgan_epoch<N>.pt`), c'est le modèle final le plus abouti. Le checkpoint SRResNet est un intermédiaire utile uniquement pour reprendre l'entraînement.

---

## Évaluation (avec images HR de référence)

Calcule le PSNR sur chaque image et génère un fichier `result/psnr_results.txt`.

```bash
python main.py \
  --mode test \
  --LR_path ./custom_dataset/hr_valid_LR_x4 \
  --GT_path ./custom_dataset/hr_valid_HR \
  --generator_path checkpoints/srgan_epoch0100.pt
```

Les images super-résolues sont sauvegardées dans `result/`.

---

## Inférence (sans images HR)

Pour upscaler des images sans avoir de référence HR :

```bash
python main.py \
  --mode test_only \
  --LR_path ./test_data/cars/ \
  --generator_path checkpoints/srgan_epoch0100.pt
```

### Utiliser les modèles pré-entraînés sur DIV2K

Les modèles `pretrained_models/SRGAN.pt` et `pretrained_models/SRResNet.pt` sont directement utilisables :

```bash
# Évaluation avec calcul PSNR
python main.py \
  --mode test \
  --LR_path ./custom_dataset/hr_valid_LR_x4 \
  --GT_path ./custom_dataset/hr_valid_HR \
  --generator_path ./pretrained_models/SRGAN.pt

# Inférence pure sur vos propres images
python main.py \
  --mode test_only \
  --LR_path ./test_data/cars/ \
  --generator_path ./pretrained_models/SRGAN.pt
```

---

## Logs et sauvegardes

| Dossier | Contenu |
|---|---|
| `logs/` | Logs TensorBoard + `run.log` (toutes les exécutions) |
| `checkpoints/` | Checkpoints complets (modèle + optimizer + epoch) |
| `result/` | Images générées + `psnr_results.txt` |

Les checkpoints permettent de **reprendre exactement** là où l'entraînement s'est arrêté via `--resume`.
