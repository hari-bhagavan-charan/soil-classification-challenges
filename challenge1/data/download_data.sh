#!/bin/bash

# Download dataset for Challenge 1 from Kaggle
KAGGLE_COMPETITION="soil-classification"
TARGET_DIR="./challenge1/data"

echo "Downloading dataset: $KAGGLE_COMPETITION"
mkdir -p "$TARGET_DIR"
kaggle competitions download -c "$KAGGLE_COMPETITION" -p "$TARGET_DIR" --unzip

echo "Download complete. Files saved to $TARGET_DIR"
