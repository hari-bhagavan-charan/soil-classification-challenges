# Importing essential libraries for data manipulation, image handling, and deep learning
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from PIL import Image

# Importing PyTorch libraries for building and training the neural network
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models

# Importing evaluation metrics from scikit-learn
from sklearn.metrics import f1_score

# Importing tqdm for displaying progress bars during loops
from tqdm import tqdm

# Importing copy for creating deep copies of objects (used for saving the best model)
import copy
# Importing time for tracking training duration
import time

# Checking if a GPU is available and setting the device accordingly (GPU for speed, otherwise CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Defining the directory paths for the training and test image datasets
train_dir = '/content/drive/MyDrive/soil-classification/soil_classification-2025/train'
test_dir = '/content/drive/MyDrive/soil-classification/soil_classification-2025/test'

# Loading the CSV files containing training labels and a test file (though this test file might be re-created later)
train_df = pd.read_csv('/content/drive/MyDrive/soil-classification/soil_classification-2025/train_labels.csv')
test_df = pd.read_csv('/content/drive/MyDrive/soil-classification/soil_classification-2025/train_labels.csv')
