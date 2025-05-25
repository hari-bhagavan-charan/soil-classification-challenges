# Importing os and pandas if not already imported (they were, but good practice to show imports with relevant sections)
import os
import pandas as pd

# Defining the path to your test image directory
test_dir = '/content/drive/MyDrive/soil-classification/soil_classification-2025/test'

# List all files in the test directory and filter for common image extensions
test_image_ids = [f for f in os.listdir(test_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

# Create a DataFrame with the image IDs found in the test directory
# This DataFrame serves as the input for your SoilDataset for inference
test_df = pd.DataFrame({'image_id': test_image_ids})

# Print the first few rows of the created test_df to verify
print("Created Test DataFrame from directory listing:")
print(test_df.head())

# Assuming SoilDataset class and test_transform are already defined
# If not, you need to define them in a preceding cell.

# Create the test dataset using the generated DataFrame and the test directory path
test_dataset = SoilDataset(test_df, test_dir, transform=test_transform, is_test=True)

# Create the DataLoader for the test dataset (shuffle=False is standard for test data)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

print("\nSuccessfully created test_dataset and test_loader from the images in the specified directory.")

# Defining a function to make predictions on the test dataset
def predict(model):
    # Set the model to evaluation mode (disables dropout and batch normalization updates)
    model.eval()
    # Lists to store the predicted class IDs and the original image IDs
    predictions = []
    image_ids = []

    # Disable gradient calculation during inference (saves memory and computation)
    with torch.no_grad():
        # Iterate through the test data in batches using the test_loader
        for images, ids in tqdm(test_loader, desc="Predicting"):
            # Move images to the specified device
            images = images.to(device)
            # Perform a forward pass to get model outputs (logits)
            outputs = model(images)
            # Get the predicted class (index with the highest logit)
            _, preds = torch.max(outputs, 1)
            # Extend the lists with predicted class IDs and image IDs
            predictions.extend(preds.cpu().numpy())
            image_ids.extend(ids)

    # Convert numerical predictions back to soil labels (strings) using the inverse label map
    pred_labels = [inv_label_map[p] for p in predictions]
    # Return a pandas DataFrame containing image IDs and their predicted soil types
    return pd.DataFrame({'image_id': image_ids, 'soil_type': pred_labels})

# Call the predict function to generate predictions on the test data
submission = predict(model)

# Ensure the output directory exists before saving the file
output_dir = 'working'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"Created output directory: {output_dir}")

# Save the initial submission CSV file
submission.to_csv(os.path.join(output_dir, 'submission.csv'), index=False)
print(" Submission file saved: submission.csv")

# Import pandas again for clarity within this section
import pandas as pd

# Define the path to the file containing the desired order of image IDs
test_ids_path = '/content/drive/MyDrive/soil-classification/soil_classification-2025/test_ids.csv'

# Load the test_ids.csv file into a DataFrame
try:
    test_ids_df = pd.read_csv(test_ids_path)
    print(f"Successfully loaded test_ids.csv from: {test_ids_path}")
    print("First 5 rows of test_ids_df:")
    print(test_ids_df.head())
except FileNotFoundError:
    # Handle the case where test_ids.csv is not found
    print(f"Error: test_ids.csv not found at {test_ids_path}")
    test_ids_df = pd.DataFrame({'image_id': []}) # Create an empty DataFrame
except Exception as e:
    # Handle other potential errors during file reading
    print(f"An error occurred while reading test_ids.csv: {e}")
    test_ids_df = pd.DataFrame({'image_id': []}) # Create an empty DataFrame

# Check if test_ids_df was loaded successfully and is not empty
if not test_ids_df.empty and 'image_id' in test_ids_df.columns:
    # Create a temporary index on test_ids_df to preserve the original order
    test_ids_df['order'] = range(len(test_ids_df))

    # Merge the generated submission DataFrame with test_ids_df
    # This aligns the submission predictions with the order specified in test_ids.csv
    # 'how='left'' ensures all IDs from test_ids_df are included
    ordered_submission = pd.merge(test_ids_df, submission, on='image_id', how='left')

    # Sort the merged DataFrame based on the temporary 'order' column
    ordered_submission = ordered_submission.sort_values(by='order')

    # Drop the temporary 'order' column as it's no longer needed
    ordered_submission = ordered_submission.drop(columns=['order'])

    # Replace the original submission DataFrame with the reordered one
    submission = ordered_submission

    print("\nSubmission DataFrame reordered based on test_ids.csv.")
    print("First 5 rows of the reordered submission DataFrame:")
    print(submission.head())

    # Check for any missing predictions after merging
    if submission['soil_type'].isnull().any():
        print("\nWarning: Some image_ids from test_ids.csv were not found in the prediction results.")
        print(submission[submission['soil_type'].isnull()])
else:
    print("\nWarning: test_ids.csv was not loaded or is empty. Cannot reorder submission.")
    print("Using the submission dataframe generated from the test directory listing.")

# Ensure the output directory exists before saving the file
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"Created output directory: {output_dir}")

# Save the final (potentially reordered) submission CSV file
submission.to_csv(os.path.join(output_dir, 'submission.csv'), index=False)
print(f"✅ Submission file saved: {os.path.join(output_dir, 'submission.csv')}")
