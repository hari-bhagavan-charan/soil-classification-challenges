'''Author:Hari bhagawan charan
Team Name: Hari bhagawan charan
Leaderboard Rank: 7 '''


''' Preprocessing + Inference Preparation Phase'''

# Download Dataset and Model
import requests, zipfile, io, os, shutil

def download_file(url, output_path):
    print(f"Downloading {output_path}...")
    response = requests.get(url, stream=True)
    with open(output_path, 'wb') as f:
        shutil.copyfileobj(response.raw, f)
    print(f"{output_path} downloaded successfully.\n")

# Download and extract dataset
dataset_url = "https://www.dropbox.com/scl/fo/uiba7fs5fqc6xgukzdrq6/AHvlDicVORTbKhLKr8H14Yc?rlkey=av3kmv5gp4qss21wjyjag449c&st=m9mgnzge&dl=1"
dataset_zip = "dataset.zip"
dataset_extract_path = "./soil_competition-2025"

download_file(dataset_url, dataset_zip)
with zipfile.ZipFile(dataset_zip, 'r') as zip_ref:
    zip_ref.extractall(dataset_extract_path)
os.remove(dataset_zip)

# Download pretrained model
model_url = "https://www.dropbox.com/scl/fi/fcfk62f84g2fnt9mff517/son.pth?rlkey=uvva0oi3qqf8kur6nx0vaar0z&st=9yxedlwy&dl=1"
model_output = "son.pth"
download_file(model_url, model_output)

# Imports and setup
import torch
import torch.nn as nn
from torchvision import transforms, models
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import timm
from tqdm import tqdm

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 8
NUM_WORKERS = 0
IMAGE_SIZE = 224

# Transform
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# Dataset
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
        return image, img_path
