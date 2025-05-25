# Soil Classification for Annam AI Hackathon - Part 2

## Description

This project is an entry for Task 2 of the Annam AI Hackathon, focusing on soil classification using unsupervised learning techniques. Specifically, we utilize a Vision Transformer (ViT) for feature extraction and an autoencoder for anomaly detection to classify images as either containing soil (label 1) or not containing soil (label 0). By leveraging transfer learning with ViT and the autoencoder's ability to detect deviations from learned patterns, we aim to achieve accurate classification even with an imbalanced dataset where training images are all labeled as soil.

## Problem Statement

The task is to develop a model that can accurately classify images as either containing soil or not, as part of the Kaggle SoilNet competition. This is a binary classification problem, but since the training dataset contains only soil images, we approach it as an anomaly detection task. The autoencoder learns to reconstruct soil image features, and images with high reconstruction errors are classified as non-soil.

## Data

The data is sourced from the Kaggle SoilNet competition and includes:

- **Training Labels**: A CSV file (train_labels.csv) containing 1,222 image IDs, all labeled as 1 (soil)
- **Test IDs**: A CSV file (test_ids.csv) containing image IDs for which predictions are required
- **Image Directories**: Training images in train directory and test images in test directory, located at `/kaggle/input/soil-classification-part-2/soil_competition-2025` in the Kaggle environment

The dataset is highly imbalanced, with all training images labeled as soil, necessitating an anomaly detection approach. Images vary in format (primarily .jpg and .jpeg, with some .webp, .png, and .gif) and size, requiring preprocessing for standardization.

## Methods

### Exploratory Data Analysis (EDA)

The notebook begins with EDA to understand the dataset:

- **Label Distribution**: Analyzes the distribution of labels, confirming all 1,222 training samples are labeled as soil (1), visualized with a count plot
- **Image Characteristics**: Examines image sizes, formats, and RGB pixel intensity distributions using histograms to understand variability and color profiles
- **Sample Visualization**: Displays sample soil images to provide qualitative insights into the data

### Feature Extraction

A pre-trained Vision Transformer (ViT) model, specifically `google/vit-base-patch16-224-in21k`, extracts features from images. The process involves:

- Resizing images to 224x224 pixels and normalizing them
- Extracting the [CLS] token embedding, resulting in a 768-dimensional feature vector per image
- Saving features in HDF5 files using the h5py library for efficient storage and retrieval

### Autoencoder for Anomaly Detection

A custom `SoilAutoencoder` is defined with the following architecture:

- **Encoder**: Reduces the 768-dimensional input to 128 dimensions through layers of 512 and 256 units with ReLU activation
- **Decoder**: Reconstructs back to 768 dimensions through layers of 256 and 512 units with ReLU activation, and a final layer with no activation

The autoencoder is trained to minimize Mean Squared Error (MSE) between input and reconstructed features. Since training data contains only soil images, the model learns to reconstruct soil features effectively, with non-soil images expected to have higher reconstruction errors.

### Training

The autoencoder is trained with:

- **Optimizer**: Adam with weight decay for regularization
- **Loss Function**: Mean Squared Error (MSE)
- **Batch Size**: 64
- **Epochs**: 20
- **Validation Split**: 20% of training data, stratified to maintain class distribution (all soil)
- **Callbacks**: Monitor training and validation loss, with a pseudo-F1 score calculated assuming all validation images are soil

### Anomaly Detection

Reconstruction errors are computed for the validation set, and a threshold is set as the mean plus two standard deviations of these errors (approximately 0.023620). Test images with errors above this threshold are classified as non-soil (0), while those below are classified as soil (1).

## Submission

Features are extracted for test images, reconstruction errors are computed, and predictions are generated based on the threshold. The predictions are saved in a `submission.csv` file, mapping image IDs to predicted labels (0 or 1), formatted for Kaggle submission.

## Results

Running the notebook produces the following metrics and visualizations:

- **Training Loss**: Decreases from 0.028203 to 0.008224 over 20 epochs, indicating effective learning
- **Validation Loss**: Decreases from 0.025777 to 0.009869, showing good generalization
- **Pseudo-F1 Score**: Stabilizes between 0.9462 and 0.9485, suggesting robust reconstruction of soil images
- **Reconstruction Error Threshold**: Set at approximately 0.023620 for anomaly detection

### Visualizations:

- Sample soil images to understand data characteristics
- Training and validation loss plots over epochs to assess convergence
- Pseudo-F1 score plot to evaluate performance stability
- Histograms of image sizes and RGB pixel intensities from EDA

These results suggest the model effectively learns to reconstruct soil image features, enabling accurate detection of non-soil images based on higher reconstruction errors. The submission file is ready for Kaggle evaluation.

### Performance Summary

| Metric                         | Description                | Value/Output                          |
| ------------------------------ | -------------------------- | ------------------------------------- |
| Training Loss                  | MSE over 20 epochs         | 0.028203 to 0.008224                  |
| Validation Loss                | MSE over 20 epochs         | 0.025777 to 0.009869                  |
| Pseudo-F1 Score                | Reconstruction performance | 0.9462–0.9485                        |
| Reconstruction Error Threshold | For anomaly detection      | ~0.023620                             |
| Visualizations                 | EDA and performance plots  | Count plot, histograms, loss/F1 plots |

## How to Run

The notebook is optimized for the Kaggle platform, leveraging pre-installed libraries and GPU support (two NVIDIA Tesla T4 GPUs).

### Running on Kaggle

1. Access the SoilNet competition page
2. Create or open the notebook (`annam-soil-classification-part-2.ipynb`) in Kaggle's editor
3. Add the competition dataset via the "Data" panel if not already included
4. Execute the notebook cell by cell, following the structured workflow from setup to submission

### Running Locally

To run locally, replicate the Kaggle environment:

#### Install Required Libraries:

Python 3, TensorFlow, PyTorch, Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, Transformers (Hugging Face), h5py, OpenCV, PIL (Pillow)

#### Configure GPU:

Install CUDA toolkit and cuDNN for GPU support

#### Download Dataset:

Obtain the SoilNet dataset from Kaggle and organize it locally

#### Update Paths:

Modify data paths (e.g., from `/kaggle/input/soil-classification-part-2/` to local directories)

#### Run Notebook:

Open in a Jupyter environment and execute cells sequentially, ensuring dependencies and paths are correctly set

**Note**: Local execution may require additional setup for GPU compatibility and library versions.

## Notebook Structure

The notebook is organized into the following sections:

1. **Setup and Imports**: Initialize environment and import necessary libraries
2. **Data Loading and EDA**: Load data, analyze label distribution, examine image characteristics, and visualize samples
3. **Feature Extraction Setup**: Configure ViT model and define feature extraction functions
4. **Feature Extraction and Storage**: Extract features from training and test images, save to HDF5 files
5. **Autoencoder Definition**: Define the SoilAutoencoder class with encoder-decoder architecture
6. **Model Training**: Train the autoencoder on soil image features with validation monitoring
7. **Anomaly Detection**: Calculate reconstruction errors and set threshold for classification
8. **Test Predictions**: Generate predictions for test images and create submission file

## Technical Details

### Key Parameters:

- **Image Size**: 224×224 pixels
- **ViT Feature Dimension**: 768
- **Autoencoder Latent Dimension**: 128
- **Batch Size**: 64
- **Training Epochs**: 20
- **Validation Split**: 20%
- **Anomaly Threshold**: Mean + 2×Standard Deviation of reconstruction errors

### Architecture Details:

- **Encoder**: 768 → 512 → 256 → 128 (ReLU activation)
- **Decoder**: 128 → 256 → 512 → 768 (ReLU activation, linear output)

### Hardware Requirements:

- **GPU**: NVIDIA GPU with CUDA support (dual Tesla T4 recommended)
- **RAM**: Minimum 16GB for local execution
- **Storage**: HDF5 files for efficient feature storage and retrieval

## Key Innovations

- **Anomaly Detection Approach**: Novel use of autoencoder for binary classification with single-class training data
- **Feature Storage Optimization**: HDF5 files for efficient feature caching and reuse
- **Threshold Selection**: Statistical approach using mean + 2σ for robust anomaly detection
- **Multi-format Image Support**: Handles various image formats (.jpg, .jpeg, .webp, .png, .gif)

## Contributors

Team participating in Annam AI Hackathon

## Additional Information

- **Kaggle Competition**: SoilNet (soil_classification-part-2)
- **Key Insight**: The use of ViT and an autoencoder highlights a modern approach to anomaly detection, suitable for imbalanced datasets
- **Performance Note**: The pseudo-F1 score is a proxy metric, as validation data contains only soil images; performance on non-soil images depends on the test set
- **Further Reading**:
  - Vision Transformers for feature extraction details
  - Autoencoders for anomaly detection theory
  - Anomaly detection in computer vision applications
