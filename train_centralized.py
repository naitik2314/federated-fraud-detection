import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split 
import numpy as np

from Data_Load_and_Prep.data_utils import load_and_preprocess_data 
from Model.model import FraudNet 
from client import test as evaluate_model # Re-use the updated test function

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Centralized training using {DEVICE}")

# --- Configuration Parameters ---
DATA_PATH = "Data/creditcard.csv" 
BATCH_SIZE = 64  
LEARNING_RATE = 0.001 
EPOCHS = 20 
SEED = 42 
TEST_SPLIT_SIZE = 0.2 

def set_seeds(seed_value):
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)
    print(f"Seeds have been set to: {seed_value} for centralized training.")

def main():
    set_seeds(SEED)

    print("Loading and preprocessing full dataset...")
    X_global, y_global = load_and_preprocess_data(csv_path=DATA_PATH)
    print(f"Full dataset loaded. X shape: {X_global.shape}, y shape: {y_global.shape}")

    X_train_central, X_test_central, y_train_central, y_test_central = train_test_split(
        X_global, y_global, 
        test_size=TEST_SPLIT_SIZE, 
        random_state=SEED, 
        stratify=y_global 
    )
    
    num_features = X_train_central.shape[1]
    print(f"Centralized training set size: {len(X_train_central)}")
    print(f"Test set size: {len(X_test_central)}")
    print(f"Number of features: {num_features}")

    tensor_x_train = torch.Tensor(X_train_central)
    tensor_y_train = torch.Tensor(y_train_central).unsqueeze(1)
    train_dataset_central = TensorDataset(tensor_x_train, tensor_y_train)
    train_loader_central = DataLoader(train_dataset_central, batch_size=BATCH_SIZE, shuffle=True)

    tensor_x_test = torch.Tensor(X_test_central)
    tensor_y_test = torch.Tensor(y_test_central).unsqueeze(1)
    test_dataset_central = TensorDataset(tensor_x_test, tensor_y_test)
    test_loader_central = DataLoader(test_dataset_central, batch_size=1024) 

    model = FraudNet(num_features=num_features).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print(f"Starting centralized model training for {EPOCHS} epochs...")
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

    print("\nEvaluating centralized model on the test set...")
    # The evaluate_model (client.test) function now takes an explicit device argument
    test_loss, test_metrics, num_test_examples = evaluate_model(model, test_loader_central, DEVICE) 
    
    print("-" * 40)
    print("Centralized Model Performance:")
    print(f"  Test Loss: {test_loss:.4f}")
    print(f"  Test Accuracy: {test_metrics.get('accuracy', -1):.4f}")
    print(f"  Test Precision: {test_metrics.get('precision', -1):.4f}")
    print(f"  Test Recall: {test_metrics.get('recall', -1):.4f}")
    print(f"  Test F1-score: {test_metrics.get('f1_score', -1):.4f}")
    print(f"  TP: {test_metrics.get('true_positives', -1)}, FP: {test_metrics.get('false_positives', -1)}")
    print(f"  TN: {test_metrics.get('true_negatives', -1)}, FN: {test_metrics.get('false_negatives', -1)}")
    print(f"  Evaluated on {num_test_examples} samples")
    print("-" * 40)

if __name__ == "__main__":
    main()