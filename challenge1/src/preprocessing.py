''' Author:Hari bhagawan charan
Team Name: Hari bhagawan charan

Leaderboard Rank: 38
'''







'''Preprocessing + Training Phase'''


# Importing libraries
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models

from sklearn.metrics import f1_score
from tqdm import tqdm
import copy
import time

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# File paths
train_dir = '/content/drive/MyDrive/soil-classification/soil_classification-2025/train'
test_dir = '/content/drive/MyDrive/soil-classification/soil_classification-2025/test'
train_df = pd.read_csv('/content/drive/MyDrive/soil-classification/soil_classification-2025/train_labels.csv')

# Label mapping
label_map = {
    'Alluvial soil': 0,
    'Black Soil': 1,
    'Clay soil': 2,
    'Red soil': 3
}
train_df['label'] = train_df['soil_type'].map(label_map)

# Dataset class
class SoilDataset(Dataset):
    def __init__(self, dataframe, root_dir, transform=None, is_test=False):
        self.df = dataframe
        self.root_dir = root_dir
        self.transform = transform
        self.is_test = is_test

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        image_id = self.df.iloc[idx, 0]
        img_path = os.path.join(self.root_dir, image_id)
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        if self.is_test:
            return image, image_id
        else:
            label = self.df.iloc[idx, -1]
            return image, label

# Transforms and data loaders
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

train_dataset = SoilDataset(train_df, train_dir, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Model setup
model = models.resnet18(pretrained=True)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 4)
model = model.to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 10
best_f1 = 0.0
best_model_wts = copy.deepcopy(model.state_dict())

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    y_true, y_pred = [], []

    for images, labels in tqdm(train_loader):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        preds = torch.argmax(outputs, dim=1)
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())

    epoch_f1 = f1_score(y_true, y_pred, average='weighted')
    print(f"Epoch {epoch+1}/{num_epochs} - Loss: {running_loss:.4f} - F1 Score: {epoch_f1:.4f}")

    if epoch_f1 > best_f1:
        best_f1 = epoch_f1
        best_model_wts = copy.deepcopy(model.state_dict())

model.load_state_dict(best_model_wts)
