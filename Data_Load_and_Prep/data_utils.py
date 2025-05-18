import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import TensorDataset, DataLoader
import numpy as np

# Default values, can be overridden when functions are called
NUM_CLIENTS_DEFAULT = 5
BATCH_SIZE_DEFAULT = 64

def load_and_preprocess_data(csv_path="Data/creditcard.csv"): # Default path for direct script run
    """
    Loads the credit card fraud dataset, preprocesses it.
    """
    df = pd.read_csv(csv_path)
    scaler = StandardScaler()
    df['scaled_amount'] = scaler.fit_transform(df['Amount'].values.reshape(-1, 1))
    df = df.drop(['Time', 'Amount'], axis=1)
    
    cols = list(df.columns)
    cols.remove('Class')
    cols.remove('scaled_amount')
    new_cols_order = cols + ['scaled_amount', 'Class']
    df = df[new_cols_order]
    
    X = df.drop('Class', axis=1).values
    y = df['Class'].values
    
    return X, y

def create_partitioned_dataloaders(
    X, y, num_clients=NUM_CLIENTS_DEFAULT, 
    batch_size=BATCH_SIZE_DEFAULT, seed=42, 
    non_iid_alpha=None 
):
    """
    Splits the data into training and testing sets, then partitions the training
    data among clients. Can create IID or non-IID (label skew) partitions.
    Creates PyTorch DataLoaders for each client and a central test DataLoader.
    """
    np.random.seed(seed) 
    torch.manual_seed(seed) 

    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )

    num_features = X_train_full.shape[1]
    client_data_loaders = []

    if non_iid_alpha is None:
        print("Creating IID-like data partitions...")
        samples_per_client = len(X_train_full) // num_clients
        for i in range(num_clients):
            start_idx = i * samples_per_client
            end_idx = (i + 1) * samples_per_client if i < num_clients - 1 else len(X_train_full)
            
            client_X_train = X_train_full[start_idx:end_idx]
            client_y_train = y_train_full[start_idx:end_idx]
            
            tensor_x_train = torch.Tensor(client_X_train)
            tensor_y_train = torch.Tensor(client_y_train).unsqueeze(1)
            
            client_dataset = TensorDataset(tensor_x_train, tensor_y_train)
            client_loader = DataLoader(client_dataset, batch_size=batch_size, shuffle=True)
            client_data_loaders.append(client_loader)
            print(f"  Client {i} (IID): {len(client_dataset)} samples. Class distribution: {dict(zip(*np.unique(client_y_train, return_counts=True)))}")

    else:
        print(f"Creating Non-IID data partitions with alpha={non_iid_alpha}...")
        num_classes = len(np.unique(y_train_full))
        
        if isinstance(X_train_full, torch.Tensor): X_train_full = X_train_full.numpy()
        if isinstance(y_train_full, torch.Tensor): y_train_full = y_train_full.numpy()

        class_indices = [np.where(y_train_full == i)[0] for i in range(num_classes)]
        client_indices = [[] for _ in range(num_clients)]
        
        for c_idx in range(num_classes):
            current_class_indices = class_indices[c_idx]
            np.random.shuffle(current_class_indices)
            
            if len(current_class_indices) == 0:
                print(f"Warning: Class {c_idx} has no samples in the training set.")
                continue

            proportions = np.random.dirichlet(alpha=np.full(num_clients, non_iid_alpha))
            samples_per_client_for_class = (proportions * len(current_class_indices)).astype(int)
            
            diff = len(current_class_indices) - samples_per_client_for_class.sum()
            for i_diff in range(diff): 
                samples_per_client_for_class[i_diff % num_clients] += 1
            
            start = 0
            for client_id in range(num_clients):
                take = samples_per_client_for_class[client_id]
                client_indices[client_id].extend(current_class_indices[start : start + take])
                start += take
        
        for i in range(num_clients):
            if not client_indices[i]: # Check if the list is empty
                client_X_train = np.array([])
                client_y_train = np.array([])
                print(f"  Client {i}: 0 samples assigned.")
            else:
                client_X_train = X_train_full[client_indices[i]]
                client_y_train = y_train_full[client_indices[i]]
                print(f"  Client {i}: {len(client_X_train)} samples. Class distribution: {dict(zip(*np.unique(client_y_train, return_counts=True)))}")

            tensor_x_train = torch.Tensor(client_X_train)
            tensor_y_train = torch.Tensor(client_y_train).unsqueeze(1) if len(client_y_train) > 0 else torch.Tensor([])
            
            client_dataset = TensorDataset(tensor_x_train, tensor_y_train)
            # Only add dataloader if there's data, and handle drop_last carefully
            if len(client_dataset) > 0 :
                client_loader = DataLoader(client_dataset, batch_size=batch_size, shuffle=True, drop_last=(len(client_dataset) % batch_size == 0 and len(client_dataset) >= batch_size) )
                client_data_loaders.append(client_loader)
            else: 
                empty_dataset = TensorDataset(torch.Tensor([]), torch.Tensor([]))
                client_loader = DataLoader(empty_dataset, batch_size=batch_size)
                client_data_loaders.append(client_loader)


    tensor_x_test = torch.Tensor(X_test)
    tensor_y_test = torch.Tensor(y_test).unsqueeze(1)
    test_dataset = TensorDataset(tensor_x_test, tensor_y_test)
    test_loader = DataLoader(test_dataset, batch_size=1024) 
    
    return client_data_loaders, test_loader, num_features

if __name__ == '__main__':
    print("--- Loading data for standalone test ---")
    X_g, y_g = load_and_preprocess_data() # Uses default "Data/creditcard.csv"
    
    print("\n--- Testing IID-like data partitioning ---")
    client_loaders_iid, test_loader_iid, n_features_iid = create_partitioned_dataloaders(
        X_g, y_g, num_clients=NUM_CLIENTS_DEFAULT, batch_size=BATCH_SIZE_DEFAULT, non_iid_alpha=None, seed=42
    )
    print(f"IID: Number of features: {n_features_iid}")
    # Class distribution printout is now inside the IID block of create_partitioned_dataloaders

    print("\n--- Testing Non-IID data partitioning (alpha=0.5) ---")
    NON_IID_ALPHA_EXAMPLE = 0.5 
    client_loaders_non_iid, test_loader_non_iid, n_features_non_iid = create_partitioned_dataloaders(
        X_g, y_g, num_clients=NUM_CLIENTS_DEFAULT, batch_size=BATCH_SIZE_DEFAULT, non_iid_alpha=NON_IID_ALPHA_EXAMPLE, seed=42
    )
    print(f"Non-IID (alpha={NON_IID_ALPHA_EXAMPLE}): Number of features: {n_features_non_iid}")
    # Class distribution printout is now inside the Non-IID block of create_partitioned_dataloaders

    print("\nCentral test DataLoader samples (from Non-IID test):", len(test_loader_non_iid.dataset))