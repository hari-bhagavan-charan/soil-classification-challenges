
# Importing the files module from google.colab for file download functionality
from google.colab import files

# Attempting to download the submission file from the specified path
try:
    files.download(os.path.join(output_dir, 'submission.csv'))
    print("Download initiated.")
except Exception as e:
    # Handle potential errors during the download process
    print(f"An error occurred during download: {e}")
