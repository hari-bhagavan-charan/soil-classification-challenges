
How to set up and run 

you need to take the code out and replace the file  paths. That's it, and use Google Colab for uninterrupted running 
As I ran the entire model on the collab

So, copy the code and paste it in the collab, change the required paths 


# Soil Type Classification using PyTorch

This project performs soil image classification using a pre-trained ResNet18 model with PyTorch.

## 📁 Dataset Structure

- `train/`: Folder with labeled training images.
- `test/`: Folder with test images.
- `train_labels.csv`: Contains image names and their corresponding soil types.
- `test_ids.csv`: Contains test image names to be predicted.

## ⚙️ Workflow

1. **Preprocessing**
   - Label encoding.
   - Train-validation split.
   - Image resizing and normalization.
   - Data augmentation with PyTorch transforms.

2. **Model Setup**
   - Transfer learning using `resnet18`.
   - Final layer modified to fit number of classes.
   - Trained on GPU if available.

3. **Training**
   - Optimizer: Adam.
   - Loss: CrossEntropyLoss.
   - Metrics: Accuracy, F1 Score.
   - Best model saved during training.

4. **Postprocessing**
   - Load best model.
   - Predict labels for test images.
   - Output predictions to `submission.csv`.



