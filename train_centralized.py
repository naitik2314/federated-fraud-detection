import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split # For consistent data splitting
import numpy as np

# Import from your project structure
from Data_Load_and_Prep.data_utils import load_and_preprocess_data # To load and preprocess
from Model.model import FraudNet # Your model definition
from client import test as evaluate_model # Re-use the test function for consistent evaluation

# Setup device
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Centralized training using {DEVICE}")

# --- Configuration Parameters ---
DATA_PATH = "Data/creditcard.csv" # Path to your CSV file
BATCH_SIZE = 64  # Batch size for centralized training
LEARNING_RATE = 0.001 # Learning rate
EPOCHS = 20 # Number of epochs for centralized training (can be tuned)
SEED = 42 # Seed for reproducibility (same as in FL setup)
TEST_SPLIT_SIZE = 0.2 # Proportion of data to use for the test set (same as in FL)

def set_seeds(seed_value):
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)
    print(f"Seeds have been set to: {seed_value} for centralized training.")

def main():
    set_seeds(SEED)

    # --- 1. Load and Prepare Full Dataset ---
    print("Loading and preprocessing full dataset...")
    X_global, y_global = load_and_preprocess_data(csv_path=DATA_PATH)
    print(f"Full dataset loaded. X shape: {X_global.shape}, y shape: {y_global.shape}")

    # --- 2. Create Centralized Training and Test Sets ---
    # We need to ensure this split is identical to the one that created
    # X_train_full and the central_test_loader in the FL setup.
    X_train_central, X_test_central, y_train_central, y_test_central = train_test_split(
        X_global, y_global, 
        test_size=TEST_SPLIT_SIZE, 
        random_state=SEED, 
        stratify=y_global # Important for imbalanced datasets
    )
    
    num_features = X_train_central.shape[1]
    print(f"Centralized training set size: {len(X_train_central)}")
    print(f"Test set size: {len(X_test_central)}")
    print(f"Number of features: {num_features}")

    # Create PyTorch DataLoaders
    # Training DataLoader
    tensor_x_train = torch.Tensor(X_train_central)
    tensor_y_train = torch.Tensor(y_train_central).unsqueeze(1)
    train_dataset_central = TensorDataset(tensor_x_train, tensor_y_train)
    train_loader_central = DataLoader(train_dataset_central, batch_size=BATCH_SIZE, shuffle=True)

    # Test DataLoader (this should be identical to central_test_loader in FL)
    tensor_x_test = torch.Tensor(X_test_central)
    tensor_y_test = torch.Tensor(y_test_central).unsqueeze(1)
    test_dataset_central = TensorDataset(tensor_x_test, tensor_y_test)
    # Use a larger batch size for testing, consistent with FL setup if possible (e.g., 1024)
    test_loader_central = DataLoader(test_dataset_central, batch_size=1024) 

    # --- 3. Initialize Model, Criterion, Optimizer ---
    model = FraudNet(num_features=num_features).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print(f"Starting centralized model training for {EPOCHS} epochs...")
    # --- 4. Training Loop ---
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        for i, (features, labels) in enumerate(train_loader_central):
            features, labels = features.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        epoch_loss = running_loss / len(train_loader_central)
        print(f"Epoch {epoch+1}/{EPOCHS}, Training Loss: {epoch_loss:.4f}")

    print("Centralized training finished.")

    # --- 5. Evaluation ---
    print("Evaluating centralized model on the test set...")
    # Use the imported evaluate_model function (which is client.test)
    test_loss, test_accuracy, num_test_examples = evaluate_model(model, test_loader_central)
    
    print("-" * 30)
    print("Centralized Model Performance:")
    print(f"  Test Loss: {test_loss:.4f}")
    print(f"  Test Accuracy: {test_accuracy:.4f} on {num_test_examples} samples")
    print("-" * 30)

if __name__ == "__main__":
    main()