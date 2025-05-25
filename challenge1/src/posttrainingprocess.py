''' Author:Hari bhagawan charan
Team Name: Hari bhagawan charan

Leaderboard Rank: 38
'''


''' Post-processing Phase (Prediction & Evaluation)'''


# Load test dataset (reusing training file, might need to update)
test_df = pd.read_csv('/content/drive/MyDrive/soil-classification/soil_classification-2025/train_labels.csv')

test_dataset = SoilDataset(test_df, test_dir, transform=transform, is_test=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

model.eval()
predictions = []

with torch.no_grad():
    for images, image_ids in tqdm(test_loader):
        images = images.to(device)
        outputs = model(images)
        preds = torch.argmax(outputs, dim=1).cpu().numpy()
        predictions.extend(zip(image_ids, preds))

# Inverse label map for readability
inv_label_map = {v: k for k, v in label_map.items()}

# Create submission dataframe
submission_df = pd.DataFrame(predictions, columns=['image_id', 'label'])
submission_df['soil_type'] = submission_df['label'].map(inv_label_map)

# View sample predictions
print(submission_df.head())
