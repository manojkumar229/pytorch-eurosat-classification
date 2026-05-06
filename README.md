# 🛰️ EuroSAT Image Classification with PyTorch

Welcome to the **EuroSAT Image Classification** repository! This project focuses on building, tuning, and evaluating a custom multi-class Convolutional Neural Network (CNN) using PyTorch to classify land use and land cover from satellite imagery.

---

## 📋 Table of Contents

* [Project Overview](#project-overview)
* [Dataset](#dataset)
* [Model Architecture](#model-architecture)
* [Key Features & Tuning](#key-features--tuning)
* [Performance & Hardware](#performance--hardware)
* [Results](#results)
* [Getting Started](#getting-started)
* [Usage](#usage)

---

## 🎯 Project Overview

Using the EuroSAT dataset, this project implements a multi-class image classification model from scratch. After facing initial overfitting—where the model learned quickly on training data but plateaued at ~50% on test data—we introduced data augmentation, regularization techniques, and tuned hyperparameters to achieve a final test accuracy of **93.8%**.

---

## 🌍 Dataset

The **EuroSAT** dataset is based on Sentinel-2 satellite images covering 13 spectral bands and consists of 10 different land use and land cover classes.

* **Total Classes:** 10 (e.g., Forest, River, Highway, Industrial, Residential, etc.)
* **Input Image Size:** $64 \times 64$ pixels
* **Splits:** 80% Training, 20% Testing (stratified split based on targets)

---

## 🏗️ Model Architecture

The custom architecture is a 3-block sequential CNN designed to handle feature extraction while keeping the network lightweight yet powerful:

```
Eurosat(
  (conv_block1): Sequential(
    (0): Conv2d(3, 20, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (1): ReLU()
    (2): Conv2d(20, 20, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (3): ReLU()
    (4): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  )
  (conv_block2): Sequential(...)
  (conv_block3): Sequential(...)
  (classifier): Sequential(
    (0): Flatten(start_dim=1, end_dim=-1)
    (1): Dropout(p=0.5, inplace=False)
    (2): Linear(in_features=1280, out_features=10, bias=True)
  )
)
```

---

## 🛠️ Key Features & Tuning

To combat the initial overfitting and achieve high accuracy, we implemented:

* **Data Augmentation:** Applied random horizontal flips and rotations (up to 20 degrees) using `torchvision.transforms` to improve generalization.
* **Regularization:** Added a 50% `Dropout` layer in the classifier along with weight decay ($1\text{e-}4$) in the optimizer.
* **Optimization:** Stochastic Gradient Descent (SGD) with a learning rate of $0.01$.

---

## ⚡ Performance & Hardware

A comparison was made between local CPU performance and cloud-based GPUs to understand the speedup for 160 epochs of training.

| Hardware | Time for 20 Epochs | Time for 160 Epochs (Projected) |
| :--- | :--- | :--- |
| **Local CPU** | ~40.0 minutes | ~320.0 minutes |
| **Tesla T4 GPU (Google Colab)** | ~6.3 minutes | **~50.8 minutes** |

---

## 📊 Results

After 160 epochs, the model converges with the following performance metrics:
* **Test Accuracy:** `93.8%`
* **Test Loss:** Low and stable across runs

The confusion matrix shows excellent separation between easily distinguishable classes, such as *Forest* and *Pasture*, with minor overlapping on more similar structural classes.

---

## 🚀 Getting Started

### Prerequisites

Ensure you have the following libraries installed:
```bash
pip install torch torchvision torchmetrics mlxtend scikit-learn numpy matplotlib tqdm
```

---

## 💻 Usage

1. **Dataset Download:** The `datasets.EuroSAT` class in PyTorch will automatically download the dataset to your specified `./data` directory.
2. **Training:** Run the training loop to initialize the `Eurosat` instance and train over the data loaders:
```python
python train.py
```
3. **Evaluation:** Use the `eval_model` function to track model loss, generate predictions, and plot the confusion matrix.
