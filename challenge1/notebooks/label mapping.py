# Creating a mapping from soil type names (strings) to unique numerical IDs (integers)
label_map = {
    'Alluvial soil': 0,
    'Black Soil': 1,
    'Clay soil': 2,
    'Red soil': 3
}

# Creating an inverse mapping from numerical IDs back to soil type names
inv_label_map = {v: k for k, v in label_map.items()}

# Adding a new column 'label' to the training DataFrame with numerical labels based on the 'soil_type' column
train_df['label'] = train_df['soil_type'].map(label_map)
