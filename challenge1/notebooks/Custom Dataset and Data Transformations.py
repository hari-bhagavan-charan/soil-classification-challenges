# Defining a custom Dataset class to handle loading images and their corresponding labels or IDs
class SoilDataset(Dataset):
    # Constructor for the dataset
    def __init__(self, dataframe, root_dir, transform=None, is_test=False):
        self.df = dataframe        # The input pandas DataFrame
        self.root_dir = root_dir   # The root directory where the images are stored
        self.transform = transform # Optional image transformations to apply
        self.is_test = is_test     # Flag to indicate if this is a test dataset (no labels needed)

    # Returns the total number of items in the dataset
    def __len__(self):
        return len(self.df)

    # Retrieves an item (image and label/ID) at a given index
    def __getitem__(self, idx):
        # Get the image ID from the DataFrame
        image_id = self.df.iloc[idx, 0]
        # Construct the full path to the image file
        img_path = os.path.join(self.root_dir, image_id)
        # Open the image and convert it to RGB format (ensuring consistency)
        image = Image.open(img_path).convert('RGB')

        # Apply transformations if defined
        if self.transform:
            image = self.transform(image)

        # Return image and ID if it's a test dataset
        if self.is_test:
            return image, image_id
        # Return image and numerical label if it's a training dataset
        else:
            label = self.df.iloc[idx, -1]  # Get the label from the 'label' column
            return image, label

# Defining image transformations to apply during training (includes data augmentation)
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),           # Resize images to 224x224 pixels
    transforms.RandomHorizontalFlip(),       # Randomly flip images horizontally for augmentation
    transforms.RandomRotation(15),           # Randomly rotate images by up to 15 degrees
    transforms.ColorJitter(0.3, 0.3, 0.3),    # Randomly change brightness, contrast, and saturation
    transforms.ToTensor(),                   # Convert image to a PyTorch tensor
    transforms.Normalize([0.485, 0.456, 0.406], # Normalize tensor using ImageNet mean and standard deviation
                         [0.229, 0.224, 0.225])
])

# Defining image transformations to apply during testing/inference (no augmentation)
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),           # Resize images to 224x224 pixels
    transforms.ToTensor(),                   # Convert image to a PyTorch tensor
    transforms.Normalize([0.485, 0.456, 0.406], # Normalize tensor using ImageNet mean and standard deviation
                         [0.229, 0.224, 0.225])
])

# Creating dataset objects for training and testing
train_dataset = SoilDataset(train_df, train_dir, transform=train_transform)
test_dataset = SoilDataset(test_df, test_dir, transform=test_transform, is_test=True)

# Creating DataLoaders to efficiently load data in batches
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True) # Shuffle training data
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False) # Do not shuffle test data
