# Download Dataset and Saved model

import requests, zipfile, io, os, shutil

# Function to download a file
def download_file(url, output_path):
    print(f"Downloading {output_path}...")
    response = requests.get(url, stream=True)
    with open(output_path, 'wb') as f:
        shutil.copyfileobj(response.raw, f)
    print(f"{output_path} downloaded successfully.\n")

# 1. Download and extract dataset.zip into ./soil_competition-2025
dataset_url = "https://www.dropbox.com/scl/fo/uiba7fs5fqc6xgukzdrq6/AHvlDicVORTbKhLKr8H14Yc?rlkey=av3kmv5gp4qss21wjyjag449c&st=m9mgnzge&dl=1"
dataset_zip = "dataset.zip"
dataset_extract_path = "./soil_competition-2025"

download_file(dataset_url, dataset_zip)

print(f"Extracting to {dataset_extract_path}...")
with zipfile.ZipFile(dataset_zip, 'r') as zip_ref:
    zip_ref.extractall(dataset_extract_path)
print("Extraction complete.\n")

# Delete the zip file after extraction
os.remove(dataset_zip)
print(f"Deleted {dataset_zip} to save space.\n")

# 2. Download son.pth
model_url = "https://www.dropbox.com/scl/fi/fcfk62f84g2fnt9mff517/son.pth?rlkey=uvva0oi3qqf8kur6nx0vaar0z&st=9yxedlwy&dl=1"
model_output = "son.pth"

download_file(model_url, model_output)

import os
import torch
import torch.nn as nn
from torchvision import transforms, models
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import timm
from tqdm import tqdm
import shutil

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)

# You can adjust these values based on your GPU/CPU resources
BATCH_SIZE = 8
NUM_WORKERS = 0
IMAGE_SIZE = 224

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

class TestImageDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.image_paths = [os.path.join(image_dir, img) for img in os.listdir(image_dir)
                            if img.lower().endswith(('.jpg', '.png', '.jpeg'))]
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, img_path  # return image and its original path

class HybridNet(nn.Module):
    def __init__(self, num_classes=2):
        super(HybridNet, self).__init__()
        self.cnn1 = timm.create_model('convnext_tiny', pretrained=True, num_classes=0)
        self.vit = timm.create_model('vit_tiny_patch16_224', pretrained=True, num_classes=0)
        self.efficientnet = timm.create_model('efficientnet_b0', pretrained=True, num_classes=0)
        self.resnet = timm.create_model('resnet50', pretrained=True, num_classes=0)

        self.feature_dim = (
            self.cnn1.num_features +
            self.vit.num_features +
            self.efficientnet.num_features +
            self.resnet.num_features
        )

        self.classifier = nn.Sequential(
            nn.Linear(self.feature_dim, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        feat_cnn1 = self.cnn1(x)
        feat_vit = self.vit(x)
        feat_eff = self.efficientnet(x)
        feat_res = self.resnet(x)
        combined = torch.cat([feat_cnn1, feat_vit, feat_eff, feat_res], dim=1)
        return self.classifier(combined)

def initialize_model(name, num_classes, pretrained=True):
    if name == 'resnet50':
        m = models.resnet50(pretrained=pretrained)
        in_feats = m.fc.in_features
        m.fc = nn.Linear(in_feats, num_classes)

    elif name == 'efficientnet_b0':
        m = timm.create_model('efficientnet_b0', pretrained=pretrained)
        in_feats = m.classifier.in_features
        m.classifier = nn.Linear(in_feats, num_classes)

    else:
        raise ValueError(f"Unknown model name: {name}")

    return m.to(DEVICE)

model1 = HybridNet(num_classes=2)
model1.load_state_dict(torch.load("son.pth", map_location=DEVICE)['model1_state_dict'])
model1.to(DEVICE)
model1.eval()

model2 = initialize_model('resnet50', num_classes=2)
model2.load_state_dict(torch.load("son.pth", map_location=DEVICE)['model2_state_dict'])
model2.to(DEVICE)
model2.eval()

test_dir = "./soil_competition-2025/test"
test_dataset = TestImageDataset(test_dir, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

import pandas as pd

# Store results
results = []

with torch.no_grad():
    for inputs, paths in tqdm(test_loader, desc="Predicting"):
        inputs = inputs.to(DEVICE)

        outputs1 = model1(inputs)
        outputs2 = model2(inputs)

        probs1 = torch.softmax(outputs1, dim=1)
        probs2 = torch.softmax(outputs2, dim=1)

        avg_probs = (probs1 + probs2) / 2
        preds = torch.argmax(avg_probs, dim=1).cpu().numpy()

        for img_path, pred in zip(paths, preds):
            image_id = os.path.basename(img_path)
            results.append({'image_id': image_id, 'label': int(pred)})

# Save to CSV
df = pd.DataFrame(results)
df.to_csv("Soil_Binary_Classification_2.csv", index=False)
print("Results saved to Soil_Binary_Classification_2.csv")

# prompt: re arrange the order of image id as given in test_ids /content/soil_competition-2025/test_ids.csv

# Load the test_ids from the CSV file
test_ids_df = pd.read_csv("/content/soil_competition-2025/test_ids.csv")

# Merge the results with the test_ids_df to reorder
# We use a left merge to ensure all image_ids from test_ids.csv are present
# And then sort by the order of the merged dataframe (which is based on test_ids_df)
merged_df = pd.merge(test_ids_df, df, on='image_id', how='left')

# Ensure the 'label' column is in the correct order based on the merged dataframe
# and select only 'image_id' and 'label' columns
ordered_df = merged_df[['image_id', 'label']]

# Save the reordered DataFrame to a new CSV file
ordered_df.to_csv("Soil_Binary_Classification_2_reordered.csv", index=False)

print("Reordered results saved to Soil_Binary_Classification_2_reordered.csv")
