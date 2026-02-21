# Clothing Classification with YOLOv8

This repository contains a complete, sequential PyTorch pipeline for classifying clothing images using a state-of-the-art **YOLOv8**. By leveraging transfer learning via the `ultralytics` library, this project use a pre-trained YOLOv8 model, fine-tunes it on a custom clothing dataset, and evaluates its performance on unseen data.

---

## 🚀 Features

* **Transfer Learning**: Utilizes `yolov8n-cls.pt` pre-trained on ImageNet to drastically reduce training time and boost accuracy for feature extraction.
* **Dynamic Head Replacement**: Automatically adjusts the classification head (`num_classes`) to match the exact number of folders in your dataset.
* **Data Augmentation**: Automatically applies random cropping, horizontal flipping, and strict ImageNet normalization during the training phase to build model robustness against varying orientations and lighting.
* **Segmented Pipeline**: Clear separation between the **Training Phase** (generating custom `.pt` weights) and the **Evaluation Phase** (testing the model on unseen data).
* **Automated Metrics**: Outputs a comprehensive classification report (Precision, Recall, F1-Score) and plots a visual Confusion Matrix using `scikit-learn` and `matplotlib`.

---

## 🛠️ Prerequisites

Ensure your Python environment (e.g., Google Colab, Jupyter Notebook, or local setup) has the required dependencies installed. You can install them via pip:

```bash
pip install ultralytics scikit-learn matplotlib
