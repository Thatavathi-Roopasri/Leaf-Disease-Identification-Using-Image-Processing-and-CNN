# Leaf-Disease-Identification-Using-Image-Processing-and-CNN

## Project Overview
This project identifies plant leaf diseases using deep learning and image processing.
It supports 15 classes across Pepper, Potato, and Tomato leaves, including healthy and diseased categories.

## Important Highlights
- Built with transfer learning using EfficientNetB0 for robust image classification.
- End-to-end workflow included: training, evaluation, confusion analysis, and web prediction.
- Flask API backend and browser UI support image upload and camera capture.
- Confidence-aware predictions include top classes and warning messages for common confusion pairs.

## Dataset At A Glance
- Total available images: 21,477
- Main dataset: 20,638 images (96.09%)
- Archive dataset: 839 images (3.91%)
- Train split: 16,504 images (79.96% of main dataset)
- Validation split: 2,070 images (10.03% of main dataset)
- Test split: 2,064 images (10.00% of main dataset)

Note: Model learning currently comes from main train/validation data. Archive data is currently available as additional test data.

## Model and Training Approach
- Image size: 224 x 224
- Multi-class classification with Softmax output
- Data augmentation: rotation, zoom, brightness, flips, shifts, and shear
- Class weighting to handle imbalance
- Multi-phase fine-tuning strategy for improved accuracy

## Key Outputs
- Trained CNN model artifact
- Confusion matrix
- Training accuracy/loss curves
- Classification report with per-class metrics

## Quick Start
```bash
pip install -r requirements.txt
python train_model.py
python app_improved.py
```

## Why This Project Matters
- Supports faster preliminary disease screening for agriculture.
- Reduces manual effort by providing instant image-based predictions.
- Improves practical usability with confidence scoring and safety warnings.