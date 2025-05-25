'''Author:Hari bhagawan charan
Team Name: Hari bhagawan charan
Leaderboard Rank: 7 '''


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

import pandas as pd
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

