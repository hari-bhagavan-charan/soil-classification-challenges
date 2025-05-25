#!/bin/bash

# Download challenge 2 dataset from Kaggle competition
KAGGLE_COMPETITION="soil-classification-part-2"
TARGET_DIR="./challenge2/data"

echo "Downloading dataset from competition: $KAGGLE_COMPETITION"
mkdir -p "$TARGET_DIR"
kaggle competitions download -c "$KAGGLE_COMPETITION" -p "$TARGET_DIR" --unzip

echo "Download complete. Files saved to $TARGET_DIR"



