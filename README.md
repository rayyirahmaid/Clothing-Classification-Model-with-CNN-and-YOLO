# Clothing Classification with Vision Transformer (ViT)

This repository contains a complete, sequential PyTorch pipeline for classifying clothing images using a state-of-the-art **Vision Transformer (ViT)**. By leveraging transfer learning via the `timm` (PyTorch Image Models) library, this project downloads a pre-trained ViT model, fine-tunes it on a custom clothing dataset, and evaluates its performance on unseen data.

---

## 🚀 Features

* **Transfer Learning**: Utilizes `vit_base_patch16_224` pre-trained on ImageNet to drastically reduce training time and boost accuracy for feature extraction.
* **Dynamic Head Replacement**: Automatically adjusts the classification head (`num_classes`) to match the exact number of folders in your dataset.
* **Data Augmentation**: Automatically applies random cropping, horizontal flipping, and strict ImageNet normalization during the training phase to build model robustness against varying orientations and lighting.
* **Segmented Pipeline**: Clear separation between the **Training Phase** (generating custom `.pth` weights) and the **Evaluation Phase** (testing the model on unseen data).
* **Automated Metrics**: Outputs a comprehensive classification report (Precision, Recall, F1-Score) and plots a visual Confusion Matrix using `scikit-learn` and `matplotlib`.

---

## 🛠️ Prerequisites

Ensure your Python environment (e.g., Google Colab, Jupyter Notebook, or local setup) has the required dependencies installed. You can install them via pip:

```bash
pip install torch torchvision timm scikit-learn matplotlib
