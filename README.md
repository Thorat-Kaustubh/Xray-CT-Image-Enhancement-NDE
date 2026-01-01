# Deep Learning for X-ray & CT-Based Industrial Non-Destructive Evaluation (NDE)

<p align="left">
  <img src="https://img.shields.io/badge/PyTorch-1.13+-red?logo=pytorch" />
  <img src="https://img.shields.io/badge/Python-3.8+-blue?logo=python" />
  <img src="https://img.shields.io/badge/Domain-Industrial%20Inspection-green" />
  <img src="https://img.shields.io/badge/Focus-CT%20%7C%20NDE%20%7C%20Deep%20Learning-orange" />
</p>

---

## Overview

This repository presents deep learning–based approaches for **X-ray and Computed Tomography (CT) data analysis** in **industrial Non-Destructive Evaluation (NDE)** applications.  
The work focuses on improving inspection reliability through **image enhancement, semantic segmentation, and 3D volumetric defect analysis** using convolutional neural networks.

The methods demonstrated here are motivated by **aerospace and industrial inspection scenarios**, where noisy CT images and large volumetric datasets pose challenges for defect detection and analysis.

---

## Problem Context

Industrial X-ray and CT inspection workflows often face:
- Low signal-to-noise ratio (SNR)
- Sensor and reconstruction artifacts
- High-resolution 3D volumes that are expensive to analyze manually

These factors reduce inspection accuracy and efficiency.  
Deep learning–based automation can significantly enhance image quality and enable scalable, consistent inspection.

---

## Project Scope

This repository covers **two closely related problem domains**:

### 1. X-ray / CT Image Denoising & Enhancement (2D)
- Improve image quality for inspection
- Suppress noise while preserving structural details
- Enable more reliable downstream analysis

### 2. 3D CT Volumetric Defect Segmentation
- Automatically identify internal defects in CT volumes
- Segment defect regions voxel-wise
- Extract quantitative defect features to support inspection insights

---

## Methodology

### Image Enhancement (2D)
- Encoder–decoder convolutional neural networks (U-Net)
- Skip connections for spatial detail preservation
- Supervised image-to-image learning
- Quantitative evaluation using PSNR and SSIM

### Volumetric Segmentation (3D)
- 3D U-Net architecture
- Patch-based training for memory-efficient GPU utilization
- Dice loss and Binary Cross-Entropy loss
- Post-processing using connected component analysis

---

## Key Techniques & Concepts

- 2D and 3D Convolutional Neural Networks (CNNs)
- X-ray / CT image preprocessing
- Noise and artifact analysis
- Semantic segmentation
- Volumetric data handling
- Quantitative evaluation and visualization

---

## Outputs

- Enhanced X-ray / CT images with reduced noise
- 2D semantic segmentation masks
- 3D volumetric defect segmentation masks
- Defect statistics (volume, shape, intensity)
- Slice-wise and volumetric visualizations

---

## Applications

- Aerospace component inspection
- Industrial CT-based defect detection
- Materials evaluation
- Automated non-destructive testing (NDT / NDE)

---

## Tech Stack

- **Python**
- **PyTorch**
- NumPy, SciPy
- OpenCV, scikit-image
- CUDA-enabled GPU training
- Matplotlib for visualization

---

## Training & Inference

The repository supports:
- GPU-accelerated training
- Mixed-precision training (AMP)
- CLI-based training and prediction
- Docker-based execution for reproducibility

---

## Notes on Implementation

This repository is based on well-established open-source implementations of U-Net and 3D U-Net architectures.  
The codebase is presented here as an **applied industrial inspection study**, focusing on X-ray and CT-based NDE use cases rather than dataset-specific benchmarks.

---

## References

- Ronneberger et al., *U-Net: Convolutional Networks for Biomedical Image Segmentation*
- Industrial X-ray and CT inspection literature
- Open-source PyTorch segmentation frameworks
