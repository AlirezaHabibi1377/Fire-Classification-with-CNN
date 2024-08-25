# Fire Classification with CNN (TinyVGG)

This project implements a Convolutional Neural Network (CNN) model based on the TinyVGG architecture to classify images as either containing fire or not. The model is built using PyTorch and was developed as part of a challenge to create a reliable fire detection system.

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgements](#acknowledgements)

- ## Project Overview
The goal of this project is to train a deep learning model that can classify images as either containing fire or not. The model is intended to assist in early fire detection, which is crucial for preventing wildfires and minimizing damage. Note: for trining the model in this project is used 25 epochs that for more accuracy should be considered more to better performance of model.

## Dataset
### Context
The dataset used in this project was created by our team during the NASA Space Apps Challenge in 2018. The challenge aimed to develop a model capable of recognizing fire in images. For more information on the context of the challenge, please visit [Our team page](#).

### Content
The dataset consists of images divided into two categories:
- **fire_images**: Contains 755 images of outdoor scenes with visible fire. Some images also contain heavy smoke.
- **non_fire_images**: Contains 244 images of nature scenes, including forests, trees, rivers, foggy forests, lakes, animals, roads, and waterfalls.

### Note
The dataset is imbalanced, with more images of fire than non-fire. It is recommended to create a validation set with an equal number of images per class (e.g., 40 images of fire and 40 images of non-fire) to avoid bias in model evaluation.

## Model Architecture
The model is based on the TinyVGG architecture, a simplified version of the VGG network. The architecture consists of two convolutional blocks followed by a fully connected layer. Here's a summary of the architecture:

## Installation

1. Clone the repository:
   ```git clone https://github.com/yourusername/fire-classification-cnn.git
cd fire-classification-cnn```

2. Create a virtual environment:
   ```python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate````

## Results
The performance of the model was evaluated using several metrics and visualizations, which are described below:

### 1. Gap Analysis
Gap Analysis was performed to evaluate the difference between the training and testing loss over epochs. This helps identify whether the model is overfitting (training loss is significantly lower than testing loss) or underfitting (both training and testing losses are high).

**Training vs Testing Loss Gap**:
- A plot was generated to visualize the gap between training and testing loss as the model trained over 25 epochs in this project.

![Training vs Testing Loss Gap](results/Gap%20Analysis.png)

### 2. Rate Analysis
Rate Analysis focused on the rate of change in loss and accuracy during training. This helps in understanding the learning dynamics and whether the model is converging.

![Rate of Change in Loss and Accuracy](path/to/your/plots/rate_loss_plot.png)

**Rate of Change in Loss and Accuracy**:
- Plots were generated to show how the training and testing loss decreased and how accuracy improved over time.

### 3. Model Performance Over Epochs
The performance of the model was tracked over each epoch to monitor how well it was learning.

**Training and Testing Accuracy**:
- Accuracy was plotted for both the training and testing sets over the epochs to visualize the model’s learning curve and final performance.

![Training vs Testing Accuracy](path/to/your/plots/accuracy_plot.png)

### 4. Plot the Loss Curves of the Model
The loss curves for both training and testing were plotted to provide a clear view of the model's convergence.

**Loss Curves**:
- The loss curves offer insights into how quickly the model is learning and whether it is suffering from overfitting or underfitting.

![Loss Curves](path/to/your/plots/loss_curves_plot.png)

### Final Performance
- **Training Loss**: The final training loss after the last epoch.
- **Testing Loss**: The final testing loss after the last epoch.
- **Training Accuracy**: The final accuracy on the training set.
- **Testing Accuracy**: The final accuracy on the testing set.

These plots and analyses collectively provide a comprehensive view of the model's performance throughout the training process.

## Saved Model

The trained model has been saved and can be reused for inference or further fine-tuning. The model is stored in the file:

**`pytorch_fire_classification_model_0.pth`**
