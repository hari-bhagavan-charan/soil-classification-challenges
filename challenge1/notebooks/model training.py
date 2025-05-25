# Loading a pre-trained EfficientNet-B0 model from torchvision
# 'pretrained=True' loads weights trained on the ImageNet dataset
model = models.efficientnet_b0(pretrained=True)

# Replacing the final classification layer to match the number of soil types (4 classes)
# The original layer outputs probabilities for 1000 ImageNet classes
model.classifier[1] = nn.Linear(model.classifier[1].in_features, 4)

# Moving the model to the specified device (GPU or CPU)
model = model.to(device)

# Defining the loss function for training (Cross Entropy Loss is common for multi-class classification)
criterion = nn.CrossEntropyLoss()

# Defining the optimizer (AdamW is a popular choice for deep learning models)
optimizer = optim.AdamW(model.parameters(), lr=1e-4)

# Defining a learning rate scheduler to reduce the learning rate during training based on a step approach
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

# Defining a function to train the model with early stopping
def train_model_with_early_stopping(model, train_loader, epochs=20, patience=5):
    # Create a deep copy of the initial model weights to save the best performing model
    best_model_wts = copy.deepcopy(model.state_dict())
    # Initialize the best performance score (tracking the minimum F1 score across classes)
    best_score = 0.0
    # Initialize the counter for early stopping
    patience_counter = 0

    # Loop through the specified number of training epochs
    for epoch in range(epochs):
        # Set the model to training mode
        model.train()
        # Initialize the running loss for the current epoch
        running_loss = 0.0
        # Lists to store all predictions and labels for calculating metrics
        all_preds = []
        all_labels = []

        # Iterate through the training data in batches using the train_loader
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            # Move images and labels to the specified device (GPU or CPU)
            images, labels = images.to(device), labels.to(device)

            # Zero the gradients of the optimizer
            optimizer.zero_grad()

            # Perform a forward pass to get model outputs
            outputs = model(images)
            # Calculate the loss between the outputs and the true labels
            loss = criterion(outputs, labels)
            # Perform a backward pass to compute gradients
            loss.backward()
            # Update model weights using the optimizer
            optimizer.step()

            # Get the predicted class (index with the highest probability)
            _, preds = torch.max(outputs, 1)
            # Accumulate the loss
            running_loss += loss.item()
            # Extend the lists with predictions and true labels for metric calculation
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        # Calculate class-wise F1-scores at the end of the epoch
        f1s = f1_score(all_labels, all_preds, average=None)
        # Find the minimum F1 score across all classes
        min_f1 = min(f1s)
        # Print epoch statistics
        print(f"Epoch {epoch+1} - Loss: {running_loss:.4f} | F1s: {f1s} | Min F1: {min_f1:.4f}")

        # Step the learning rate scheduler (potentially reducing the LR)
        scheduler.step()

        # Check if the current minimum F1 score is better than the best recorded score
        if min_f1 > best_score:
            # Update the best score
            best_score = min_f1
            # Save the current model weights as the new best weights
            best_model_wts = copy.deepcopy(model.state_dict())
            # Reset the early stopping counter
            patience_counter = 0
            print(" New best model saved!")
        else:
            # Increment the early stopping counter if performance did not improve
            patience_counter += 1
            print(f" No improvement. Patience: {patience_counter}/{patience}")
            # Check if the patience limit has been reached
            if patience_counter >= patience:
                print(" Early stopping triggered.")
                # Stop the training loop
                break

    # Load the best performing model weights back into the model
    model.load_state_dict(best_model_wts)
    # Return the trained model with the best weights
    return model
  # Call the training function to train the model
model = train_model_with_early_stopping(model, train_loader, epochs=20, patience=4)

