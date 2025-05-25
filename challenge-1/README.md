# Soil Classification for Annam AI Hackathon

## Problem Statement

The task is to develop a model that can accurately classify soil types based on provided images. This is a multi-class classification problem with four classes: Alluvial soil, Black Soil, Clay soil, and Red soil. The project is part of the Kaggle competition "soil_classification-2025", where the goal is to predict soil types for a test set and submit the results.

## Data

The data for this project is sourced from the Kaggle competition "soil_classification-2025". It includes:

- **Training Labels**: A CSV file containing image IDs and their corresponding soil types.
- **Test IDs**: A CSV file containing image IDs for which predictions need to be made.

The dataset may exhibit class imbalance, which is addressed in the project to ensure fair model performance across all classes.

## Methods

### Feature Extraction

We use a pre-trained Vision Transformer (ViT) model, specifically "google/vit-base-patch16-224-in21k", to extract features from the soil images. The images are preprocessed by resizing them to 224x224 pixels and normalizing them before feeding into the ViT model.

### Model Architecture

On top of the extracted features, a dense neural network is built consisting of:

- Input layer (768 features from ViT)
- Dense layers with ReLU activation (256, 128 neurons)
- Dropout layers for regularization (0.5 dropout rate)
- Output layer with softmax activation for multi-class classification (4 classes)

### Handling Class Imbalance

The dataset exhibits class imbalance, which is addressed by computing class weights and incorporating them into the model training process to ensure balanced learning across all classes.

### Training

The model is trained using the Adam optimizer (learning rate: 0.0001) and categorical crossentropy loss. Training is monitored with callbacks for early stopping (patience=10) and learning rate reduction (factor=0.5, patience=3) to prevent overfitting.

## Evaluation

The model's performance is evaluated on a validation set (20% stratified split), with metrics including:

- **Accuracy**: Overall classification accuracy
- **F1-scores**: F1-scores for each soil type
- **Confusion Matrix**: Visualization of classification errors across soil types

## Submission

The final step involves generating predictions for the test set and creating a submission file in the required format for the Kaggle competition.

## Results

When the notebook is run, it generates the following evaluation metrics:

- **Training and Validation Plots**: Loss and accuracy plots over epochs to visualize model convergence
- **F1-Scores**: F1-scores for each soil type, indicating the model's performance on the validation set
- **Confusion Matrix**: A heatmap visualizing classification errors across all soil types

These metrics help in understanding the model's strengths and weaknesses and can be used to fine-tune the approach further. Additionally, the notebook produces a submission file ready for uploading to the Kaggle competition.

## How to Run

This notebook is designed to run on the Kaggle platform, which provides a convenient environment with pre-installed libraries and GPU support.

### Running on Kaggle

1. Go to the Kaggle competition page for "soil_classification-2025"
2. Click on "Notebook" to create a new notebook or open this existing notebook
3. Ensure that the dataset is added to your notebook by checking the "Data" panel on the right and adding the competition dataset if necessary
4. Run the notebook cell by cell. The notebook is structured to guide you through the process, from setup to submission

### Running Locally

To run this notebook locally, you need to set up a similar environment:

#### Install Required Libraries:

Python 3, TensorFlow, PyTorch, Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, Transformers (for ViT), OpenCV, PIL (Pillow)

#### Set Up GPU:

- Ensure you have a GPU with CUDA support installed
- Install the necessary CUDA toolkit and cuDNN

#### Download the Dataset:

- Download the competition dataset from Kaggle
- Place the data in a directory and adjust the paths in the notebook accordingly

#### Adjust Paths:

In the notebook, the data paths are set for Kaggle's environment (e.g., `/kaggle/input/soil-classification-2025/`). You need to change these to match your local directory structure.

#### Run the Notebook:

- Open the notebook in a Jupyter environment
- Run each cell sequentially, making sure that all dependencies are met and paths are correctly set

**Note**: Running locally might require additional setup, especially for GPU support and ensuring all libraries are compatible with your system.

## Notebook Structure

The notebook is organized into the following sections:

1. **Setup and Imports**: Initializes the environment and imports necessary libraries
2. **Data Loading and Preprocessing**: Loads the training and test data, handles missing files, and maps soil types to numeric labels
3. **Exploratory Data Analysis (EDA)**: Analyzes the class distribution and calculates the imbalance ratio
4. **Handle Class Imbalance**: Computes class weights and splits the data into training and validation sets
5. **ViT Feature Extraction**: Uses a pre-trained ViT model to extract features from the images
6. **Model Definition and Training**: Defines and trains a dense neural network on the extracted features
7. **Evaluation**: Evaluates the model on the validation set, plots loss/accuracy, calculates F1-scores, and displays a confusion matrix
8. **Test Set Predictions and Submission**: Generates predictions for the test set and creates a submission file

## Technical Details

### Key Parameters:

- **Image Size**: 224×224 pixels
- **Batch Size**: 16
- **Number of Classes**: 4
- **ViT Feature Dimension**: 768
- **Training Epochs**: 100 (with early stopping)
- **Validation Split**: 20% (stratified)

### Hardware Requirements:

- **GPU**: NVIDIA GPU with CUDA support (Tesla T4 or better recommended)
- **RAM**: Minimum 8GB (16GB recommended for local execution)

## Contributors

Team : GreenLanterns

Team Members: Sparsh Karna, Akshat Majila, Mohnish Kodukulla , Arkita B, Mishti Mattu

## Additional Information

- **Kaggle Competition**: soil_classification-2025
- **Further Reading**: For more on Vision Transformers, refer to resources like the original ViT paper or the Hugging Face Transformers Documentation
- **Model Approach**: Transfer learning with frozen ViT features + trainable dense classifier
- **Key Innovation**: Efficient batch processing for memory management during feature extraction
