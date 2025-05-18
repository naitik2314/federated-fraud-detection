import flwr as fl
import torch
from collections import OrderedDict
import numpy as np # For seeding if not already imported via other means

# Import from your project structure
from Data_Load_and_Prep.data_utils import load_and_preprocess_data, create_partitioned_dataloaders
from Model.model import FraudNet 
from client import FraudDetectionClient, test as client_test_utility 

# Setup device (globally for orchestrator, client.py handles its own DEVICE)
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Simulation orchestrator using {DEVICE}")

# --- Configuration Parameters ---
NUM_CLIENTS = 5
BATCH_SIZE = 64
NUM_ROUNDS = 10  
LOCAL_EPOCHS = 2 
LEARNING_RATE = 0.001
DATA_PATH = "Data/creditcard.csv" # Using forward slash
SEED = 42 # Seed for reproducibility

# Non-IID data configuration
NON_IID_ALPHA = 0.5  # Use a value like 0.1, 0.5 for skewed data. Set to None for IID-like split.
# NON_IID_ALPHA = None # Uncomment for IID-like split

def set_seeds(seed_value):
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)

def main():
    set_seeds(SEED) # Set seeds at the beginning of main
    print(f"Seeds have been set to: {SEED}")

    print("Loading and preprocessing data...")
    X_global, y_global = load_and_preprocess_data(csv_path=DATA_PATH)
    print(f"Data loaded. X shape: {X_global.shape}, y shape: {y_global.shape}")

    print(f"Creating partitioned DataLoaders for {NUM_CLIENTS} clients...")
    client_train_dataloaders, central_test_loader, num_features = create_partitioned_dataloaders(
        X_global, y_global, 
        num_clients=NUM_CLIENTS, 
        batch_size=BATCH_SIZE,
        seed=SEED, # Pass seed to data partitioning
        non_iid_alpha=NON_IID_ALPHA 
    )
    print(f"Number of features for the model: {num_features}")

    def model_fn():
        # Ensure model initialization is consistent if needed, though random init is often fine
        # torch.manual_seed(SEED) # Optionally re-seed before each model instantiation if strict consistency needed here
        return FraudNet(num_features=num_features)

    def client_fn(cid: str) -> fl.client.Client:
        client_id = int(cid)
        if client_id >= len(client_train_dataloaders):
            raise ValueError(f"Client ID {cid} is out of bounds for the number of dataloaders provided.")

        trainloader = client_train_dataloaders[client_id]
        # Client evaluates on its own training data partition in this setup
        valloader = trainloader 
        
        # Check if the client has any data before creating the client object
        if len(trainloader.dataset) == 0:
            print(f"Client {cid} has no data. Creating a non-participating client or special handling needed.")
            # Depending on Flower version and strategy, this client might cause issues or be skipped.
            # For now, it will instantiate but report 0 samples trained.
            # A more robust way is to filter out such clients from `num_clients` before simulation
            # or ensure min_samples_per_client in data_utils.py
        
        return FraudDetectionClient(model_fn=model_fn, trainloader=trainloader, valloader=valloader).to_client()

    def get_evaluate_fn(test_loader, model_instantiation_fn):
        def evaluate(server_round: int, parameters: fl.common.NDArrays, config: dict):
            if server_round == 0 and not parameters:
                 return None, {}

            model = model_instantiation_fn() 
            
            # Ensure model parameters are properly loaded
            if parameters:
                params_dict = zip(model.state_dict().keys(), parameters)
                state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
                model.load_state_dict(state_dict, strict=True)
            model.to(DEVICE) 

            loss, accuracy, num_examples = client_test_utility(model, test_loader)
            print(f"Round {server_round}: Server-side evaluation - Loss: {loss:.4f}, Accuracy: {accuracy:.4f} on {num_examples} test samples")
            return loss, {"accuracy": accuracy}
        return evaluate

    evaluate_fn_for_server = get_evaluate_fn(central_test_loader, model_fn)

    def fit_config(server_round: int):
        config = {
            "local_epochs": LOCAL_EPOCHS,
            "learning_rate": LEARNING_RATE,
            "server_round": server_round, 
        }
        return config
    
    # Initialize model parameters for the strategy
    # set_seeds(SEED + 1) # Optionally change seed slightly for initial model if desired, or keep global SEED
    initial_model = model_fn()
    initial_parameters = fl.common.ndarrays_to_parameters(
        [val.cpu().numpy() for _, val in initial_model.state_dict().items()]
    )

    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0,  
        fraction_evaluate=0.0, # Disable client-side evaluation sampling by FedAvg if only doing centralized
        min_fit_clients=NUM_CLIENTS,  
        min_evaluate_clients=0, # No client-side FedAvg evaluation
        min_available_clients=NUM_CLIENTS, 
        evaluate_fn=evaluate_fn_for_server, 
        on_fit_config_fn=fit_config,  
        initial_parameters=initial_parameters, 
    )

    print(f"Starting Federated Learning simulation for {NUM_ROUNDS} rounds with {NUM_CLIENTS} clients...")
    print(f"Using Non-IID Alpha: {NON_IID_ALPHA}")
    
    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=NUM_CLIENTS,
        config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
        strategy=strategy,
        client_resources={"num_cpus": 1, "num_gpus": 0.0} if DEVICE == torch.device("cpu") else {"num_cpus": 1, "num_gpus": 0.2}
    )

    print("Federated Learning simulation finished.")
    print("History (server-side evaluation metrics):", history.metrics_centralized)
    if history.losses_centralized:
         print("History (server-side evaluation loss):", history.losses_centralized)


if __name__ == "__main__":
    main()