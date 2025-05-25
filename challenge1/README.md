How to set up and run 

You need to take the code out and replace the file  paths. That's it, and use Google Colab for uninterrupted running 
As I ran the entire model on the collab

So, copy the code and paste it in the collab, change the required paths 

# Soil Type Classification using TensorFlow

This project uses a Convolutional Neural Network (CNN) built with TensorFlow and Keras to classify images of soil into different types.

## 📁 Dataset Structure

- `train/`: Folder containing labeled training images.
- `test/`: Folder containing test images (unlabeled).
- `train_labels.csv`: CSV file with image filenames and their corresponding soil labels.
- `test_ids.csv`: CSV file with test image filenames (for prediction output).

## ⚙️ Key Steps

1. **Data Preprocessing**
   - Image resizing and normalization.
   - Label encoding.
   - Data splitting into training and validation sets.

2. **Model Architecture**
   - Based on CNN layers with dropout for regularization.
   - Optimized using Adam optimizer.
   - Trained with categorical crossentropy loss.

3. **Evaluation**
   - Accuracy and F1 Score used as metrics.
   - Best model saved during validation.
     
4. **Prediction**
   - Test images loaded and passed through the model.
   - Output predictions saved to `submission.csv`.


