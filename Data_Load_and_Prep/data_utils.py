import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import TensorDataset, DataLoader

data_path = 'Data/creditcard.csv'
NUM_CLIENTS = 5
BATCH_SIZE = 64

# --- Load Credit card fraud data, preprocess it, and split it ---
def load_and_preprocess_data(csv_path = data_path):
    df = pd.read_csv(csv_path)

    # Scaling the amount, as it is vastly diff than the dataset features V1 to V28, this will help model converge faster
    scaler = StandardScaler()
    df['scaled_amount'] = scaler.fit_transform(df['Amount'].values.reshape(-1, 1))

    # Dropping unnecessary columns
    df = df.drop(['Time', 'Amount'], axis=1)

    # Move 'Class' to the end, and 'scaled_amount' to where 'Amount' was
    cols = list(df.columns)
    cols.remove('Class')
    cols.remove('scaled_amount')
    new_cols_order = cols + ['scaled_amount', 'Class']
    df = df[new_cols_order]

    X = df.drop('Class', axis=1).values
    y = df['Class'].values
    
    return X, y

# --- Split the data into training and testing sets, further partition training into clients ---
def create_partitioned_dataloaders(X, y, num_clients=NUM_CLIENTS, batch_size=BATCH_SIZE, seed=42):

    # Global training and test dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )

    client_data_loaders = []
    # Calculate samples per client
    samples_per_client = len(X_train) // num_clients

    for i in range(num_clients):
        start_idx = i * samples_per_client
        # Ensure the last client gets all remaining data
        end_idx = (i + 1) * samples_per_client if i < num_clients - 1 else len(X_train)
        
        client_X_train = X_train[start_idx:end_idx]
        client_y_train = y_train[start_idx:end_idx]
        
        # Convert to PyTorch tensors
        tensor_x_train = torch.Tensor(client_X_train)
        tensor_y_train = torch.Tensor(client_y_train).unsqueeze(1) # Target for BCEWithLogitsLoss
        
        client_dataset = TensorDataset(tensor_x_train, tensor_y_train)
        client_loader = DataLoader(client_dataset, batch_size=batch_size, shuffle=True)
        client_data_loaders.append(client_loader)

    # Create a central test DataLoader
    tensor_x_test = torch.Tensor(X_test)
    tensor_y_test = torch.Tensor(y_test).unsqueeze(1)
    test_dataset = TensorDataset(tensor_x_test, tensor_y_test)
    # Use a larger batch size for testing or full batch if memory allows
    test_loader = DataLoader(test_dataset, batch_size=1024) 
    
    # We also need the number of features for model input size
    num_features = X_train.shape[1]

    return client_data_loaders, test_loader, num_features

if __name__ == '__main__':
    
    print("Loading and preprocessing data...")
    X_global, y_global = load_and_preprocess_data()
    print(f"Data loaded. X shape: {X_global.shape}, y shape: {y_global.shape}")
    
    print(f"\nCreating partitioned DataLoaders for {NUM_CLIENTS} clients...")
    client_loaders, central_test_loader, n_features = create_partitioned_dataloaders(
        X_global, y_global, num_clients=NUM_CLIENTS, batch_size=BATCH_SIZE
    )
    
    print(f"Number of features: {n_features}")
    print(f"Number of client DataLoaders: {len(client_loaders)}")
    for i, loader in enumerate(client_loaders):
        print(f"  Client {i+1}: {len(loader.dataset)} samples, {len(loader)} batches")
    
    print(f"Central test DataLoader: {len(central_test_loader.dataset)} samples, {len(central_test_loader)} batches")

    # Inspect a batch from the first client
    for client_x_batch, client_y_batch in client_loaders[0]:
        print(f"\nSample batch from Client 1:")
        print(f"  X batch shape: {client_x_batch.shape}")
        print(f"  y batch shape: {client_y_batch.shape}")
        break