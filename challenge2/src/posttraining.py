'''Author:Hari bhagawan charan
Team Name: Hari bhagawan charan
Leaderboard Rank: 7 '''

'''Post-processing Phase (Prediction & Output)'''

# Load the model
model = timm.create_model("resnet18", pretrained=False, num_classes=4)
model.load_state_dict(torch.load("son.pth", map_location=DEVICE))
model.to(DEVICE)
model.eval()

# Predict
test_dir = "./soil_competition-2025/test"
test_dataset = TestImageDataset(test_dir, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

predictions = []

with torch.no_grad():
    for images, paths in tqdm(test_loader):
        images = images.to(DEVICE)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        predictions.extend(zip(paths, preds.cpu().numpy()))

# Output
for path, label in predictions[:10]:  # Display sample
    print(f"Image: {os.path.basename(path)} -> Predicted Label: {label}")
