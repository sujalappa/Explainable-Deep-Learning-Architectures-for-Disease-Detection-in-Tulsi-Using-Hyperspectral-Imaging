# Explainable Deep Learning Architectures for Early Disease Detection in Tulsi (*Ocimum tenuiflorum*) Using Hyperspectral Imaging

## Overview

This research develops and compares **custom deep learning architectures** specifically designed for hyperspectral image classification, applied to early disease detection in Tulsi (Holy Basil). Unlike traditional RGB-based plant disease detection, we leverage **168 spectral bands** capturing reflectance from visible to near-infrared wavelengths, enabling detection of disease markers invisible to the naked eye.

Every model includes **Explainable AI (XAI)** to provide transparent, biologically interpretable predictions — identifying *which spatial pixels* and *which spectral wavelengths* drive each disease classification decision.

## Key Contributions

- **Novel HSI-specific architectures** (HSI-ResNet, SSViT, CNN) that process hyperspectral data natively without lossy compression or artificial upsampling
- **Spectral Attention mechanisms** (Squeeze-and-Excitation blocks) that dynamically learn which wavelengths are most diagnostic for disease detection
- **Standardized 6-panel XAI dashboard** across all models using Attention Rollout (Transformers) and Gradient Saliency (CNNs)
- **Comprehensive comparison** of 5+ architectures on the same dataset under identical conditions

## Dataset

| Property | Value |
|----------|-------|
| **Plant** | Tulsi (*Ocimum tenuiflorum*) |
| **Imaging** | Hyperspectral (168 bands) |
| **Patch Size** | 5 × 5 × 168 (spatial × spectral) |
| **Classes** | Fresh (0), Disease (1), Black (2) |
| **Extraction** | Non-overlapping, foreground-only via Otsu segmentation |

## Preprocessing Pipeline

```
Raw HSI Cubes (.hdr/.bil)
    → Load via spectral library (H × W × 168)
    → Best-band selection (max std dev)
    → Otsu thresholding + morphological cleaning
    → 5×5 non-overlapping patch extraction (leaf-only)
    → StandardScaler normalization
    → X.npy (N, 5, 5, 168) + y.npy (N,)
```

## Models Implemented

### Baseline (Adapted Standard Architectures)
| Model | Approach | Limitation |
|-------|----------|------------|
| **Original ViT** | CLS token, standard positional embedding | Generic, not HSI-aware |
| **Original ResNet** | 168→3 channel compression + 5→40 upsampling | Massive spectral info loss |
| **3D-CNN** | 3D convolutions on (168, 5, 5, 1) | No spectral attention |

### Novel (HSI-Specific — Our Contribution)
| Model | Key Innovation |
|-------|---------------|
| **SSViT** | No CLS token, direct spectral-spatial patch mapping, 4 Transformer layers, 8 attention heads, label smoothing |
| **HSI-ResNet** | Spectral Attention (SE) + Dual-Path: 1D spectral residual blocks + 2D spatial residual blocks |

## Explainable AI (XAI)

Every model generates a **6-panel diagnostic dashboard**:

| Panel | Information |
|-------|-------------|
| Original Patch | RGB-approximated 5×5 visualization |
| Saliency Heatmap | Pixel-level importance (which leaf regions matter) |
| Pixel Importance Bar | All 25 pixels ranked by influence |
| Spectral Signature | Mean reflectance curve across 168 bands |
| Text Summary | Key metrics, top pixels, interpretation |
| Class Probabilities | Confidence per class |

**XAI Methods:**
- **Attention Rollout** — For Vision Transformers (traces attention flow across layers)
- **Gradient Saliency** — For CNNs/ResNets (∂Output/∂Input sensitivity)
- **t-SNE Latent Space** — For Autoencoder (cluster visualization)

## Repository Structure

```
├── extract_patches.py                   # Preprocessing: HSI → 5×5×168 patches
├── train_all_models_comparison.ipynb     # All 5 models + XAI + comparison
├── visualize_patches.ipynb               # Patch visualization
└── output_dataset_patches_3d/            # Preprocessed dataset
    ├── X.npy                             # Patches (N, 5, 5, 168)
    ├── y.npy                             # Labels (N,)
    └── meta.npy                          # Source traceability
```