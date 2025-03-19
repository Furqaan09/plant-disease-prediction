# 🌿 Plant Disease Prediction (Deep Learning Project)

A deep learning-powered web application for predicting plant diseases from leaf images using a Convolutional Neural Network (CNN). Built with TensorFlow and Streamlit for seamless deployment.

## 🚀 Features

- CNN-Based Classifier trained on the PlantVillage dataset with 38 - plant disease classes.
- Streamlit Web App for user-friendly image upload and prediction.
- Real-time Predictions with detailed output showing Plant name and Disease.
- Git LFS Support to manage large model files (>100MB).

## 🧠 Model Details

- **Architecture:** 3 Convolutional layers + MaxPooling + Dense layers with Dropout.
- **Input Size:** 224×224 RGB Images.
- **Accuracy:** ~94.5% validation accuracy.
- **Dataset:** [PlantVillage](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset) — 43,456 training images, 10,849 validation images across 38 plant disease categories.
*Note:* This model is trained **only on the color dataset** (RGB images). Grayscale or segmented images were not used.
- **Data Augmentation:** Rotation, Zoom, Horizontal Flip, Shear, Width/Height Shift for overfitting reduction.

## 📦 Getting Started

1. Clone the Repository

    **Note:** This repo uses Git LFS. Install Git LFS first.
    ```shell
    git lfs install

    git clone https://github.com/Furqaan09/plant-disease-prediction.git
    ```

2. Install the requirements

    ```shell
    cd plant-disease-prediction/app

    pip install -r requirements.txt
    ```
3. Run the Streamlit App

    ```shell
    streamlit run main.py
    ```
